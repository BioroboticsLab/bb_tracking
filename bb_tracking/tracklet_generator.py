import numba
import numba.typed
import numpy as np
import hungarian

from . import types

def calculate_detection_pair_features(open_detections,
                          frame_detections, frame_kdtree,
                          max_distance, detection_feature_fn, detection_cost_fn, n_features):
    tracklet_indices = []
    detection_indices = []

    for i, detection in enumerate(open_detections):
        x, y = detection.x_hive, detection.y_hive
        neighbors = frame_kdtree.query_ball_point((x, y), max_distance)

        for j in neighbors:
            tracklet_indices.append(i)
            detection_indices.append(j)

    all_features = np.nan * np.zeros(shape=(len(tracklet_indices), n_features), dtype=np.float32)
    for idx, (i, j) in enumerate(zip(tracklet_indices, detection_indices)):
        pair_features = detection_feature_fn(open_detections[i], frame_detections[j])
        all_features[idx] = pair_features

    return tracklet_indices, detection_indices, all_features

def generate_random_track_id():
    """Returns a unique ID that is 64 bits long.

    Taken from the bb_pipeline codebase.
    """
    import hashlib, uuid

    hasher = hashlib.sha1()
    hasher.update(uuid.uuid4().bytes)
    hash = int.from_bytes(hasher.digest(), byteorder='big')
    # strip to 64 bits
    hash = hash >> (hash.bit_length() - 64)
    return hash

def fill_distance_matrix(detection0_indices, detection1_indices, distances, matrix):
    matrix[detection0_indices, detection1_indices] = distances

class TrackletGenerator():

    def __init__(self, cam_id, detection_feature_fn, detection_cost_fn, n_features,
                 max_distance_per_second=10.0, max_seconds_gap=0.2, max_cost=1.0):
        self.cam_id = cam_id

        self.n_features = n_features
        self.detection_feature_fn = detection_feature_fn
        self.detection_cost_fn = detection_cost_fn

        self.open_tracklets = []
        self.open_tracklets_first_begin = None

        self.last_frame_datetime = None

        self.max_cost = max_cost
        self.max_distance_per_second = max_distance_per_second
        self.max_seconds_gap = max_seconds_gap
    
    def get_first_open_begin_datetime(self):
        return self.open_tracklets_first_begin

    def get_last_frame_datetime(self):
        return self.last_frame_datetime

    def finalize_all(self):
        yield from self.open_tracklets
        self.open_tracklets = []
        self.open_tracklets_first_begin = None

    def push_detections_as_new_tracklets(self, detections, frame_id, frame_datetime):
        for detection in detections:
            if self.open_tracklets_first_begin is None or frame_datetime < self.open_tracklets_first_begin:
                self.open_tracklets_first_begin = frame_datetime
            self.open_tracklets.append(types.Track(generate_random_track_id(), self.cam_id,
                                             [detection], [frame_datetime], [frame_id], None, dict()))

    def push_frame(self, frame_id, frame_datetime, frame_detections, frame_kdtree):
        
        allowed_max_distance = self.max_distance_per_second
        seconds_since_last_frame = 0.0

        # Enfore maximum gap duration (even if just one frame apart).
        if self.last_frame_datetime is not None:
            seconds_since_last_frame = (frame_datetime - self.last_frame_datetime).total_seconds()
            if seconds_since_last_frame > self.max_seconds_gap:
                yield from self.finalize_all()
            else:
                allowed_max_distance = self.max_distance_per_second * seconds_since_last_frame

        self.last_frame_datetime = frame_datetime
        n_open_tracklets = len(self.open_tracklets)
        n_new_detections = len(frame_detections)

        # Shortcut - no open tracklets yet?
        if n_open_tracklets == 0:
            self.push_detections_as_new_tracklets(frame_detections, frame_id, frame_datetime)
            return

        detection0_indices, detection1_indices, all_features = calculate_detection_pair_features(
                                            [tracklet.detections[-1] for tracklet in self.open_tracklets],
                                            frame_detections, frame_kdtree, max_distance=allowed_max_distance,
                                            detection_feature_fn=self.detection_feature_fn, n_features=self.n_features,
                                            detection_cost_fn=self.detection_cost_fn)

        if len(detection0_indices) > 0:
            distances = self.detection_cost_fn(all_features)
            detection0_indices = np.array(detection0_indices, dtype=np.int32)
            detection1_indices = np.array(detection1_indices, dtype=np.int32)

            
            square_dimension = max(n_open_tracklets, n_new_detections)
            cost_matrix = np.zeros(shape=(square_dimension, square_dimension), dtype=np.float32) + 100000.0

            fill_distance_matrix(detection0_indices, detection1_indices, distances, cost_matrix)
            lap_results = hungarian.lap(cost_matrix.copy())
            tracklet_indices, detection_indices = tuple(range(cost_matrix.shape[0])), lap_results[0]
        else:
            tracklet_indices, detection_indices = tuple(), tuple()

        linked_tracklet_indices = set()
        linked_detection_indices = set()
        for (tracklet_idx, detection_idx) in zip(tracklet_indices, detection_indices):
            cost = cost_matrix[tracklet_idx, detection_idx]
            if cost > self.max_cost:
                continue
            tracklet = self.open_tracklets[tracklet_idx]
            assert frame_datetime > tracklet.timestamps[-1]
            tracklet.detections.append(frame_detections[detection_idx])
            tracklet.timestamps.append(frame_datetime)
            tracklet.frame_ids.append(frame_id)
            tracklet.cache_.clear()

            assert tracklet_idx not in linked_tracklet_indices
            assert detection_idx not in linked_detection_indices
            
            linked_tracklet_indices.add(tracklet_idx)
            linked_detection_indices.add(detection_idx)

        # Uncontinued tracklets are closed.
        old_open_tracklets = self.open_tracklets
        self.open_tracklets = []
        self.open_tracklets_first_begin = None
        for idx, tracklet in enumerate(old_open_tracklets):
            if idx in linked_tracklet_indices:
                self.open_tracklets.append(tracklet)

                if self.open_tracklets_first_begin is None or tracklet.timestamps[0] < self.open_tracklets_first_begin:
                    self.open_tracklets_first_begin = tracklet.timestamps[0]
            else:
                yield tracklet

        # Unassigned detections become new tracklets.
        self.push_detections_as_new_tracklets((detection for idx, detection in enumerate(frame_detections) if not (idx in linked_detection_indices)),
                                              frame_id, frame_datetime)