import numba
import numpy as np
import scipy.optimize

from . import types

def get_detection_features(detection0_index, detection1_index, 
                           detection0, detection1, seconds_passed, detection_cost_fn):
    
    return detection0_index, detection1_index, detection_cost_fn(detection0, detection1, seconds_passed)

def match_detection_lists(open_detections,
                          frame_detections, frame_kdtree,
                          max_distance, **kwargs):
    for i, detection in enumerate(open_detections):
        x, y = detection.x_hive, detection.y_hive
        neighbors = frame_kdtree.query_ball_point((x, y), max_distance)

        for j in neighbors:
            yield get_detection_features(i, j, detection, frame_detections[j], **kwargs)

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

@numba.njit
def fill_distance_matrix(candidates, matrix):
    for i, j, distance in candidates:
        matrix[i, j] = distance

class TrackletGenerator():

    def __init__(self, cam_id, detection_cost_fn, max_distance_per_second=10.0, max_seconds_gap=1.0, max_cost=1.0):
        self.cam_id = cam_id

        self.detection_cost_fn = detection_cost_fn

        self.open_tracklets = []

        self.last_frame_datetime = None

        self.max_cost = max_cost
        self.max_distance_per_second = max_distance_per_second
        self.max_seconds_gap = max_seconds_gap
        self.minimum_open_tracklet_begin = None
    
    def get_minimum_open_tracklet_begin(self):
        return self.minimum_open_tracklet_begin

    def finalize_all(self):
        yield from self.open_tracklets
        self.open_tracklets = []

    def push_detections_as_new_tracklets(self, detections, frame_id, frame_datetime):
        for detection in detections:
            self.open_tracklets.append(types.Track(generate_random_track_id(), self.cam_id,
                                             [detection], [frame_datetime], [frame_id], None))

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

        cost_matrix = np.ones(shape=(n_open_tracklets, n_new_detections), dtype=np.float32) + self.max_cost + 1e10

        candidates = match_detection_lists((tracklet.detections[-1] for tracklet in self.open_tracklets),
                                            frame_detections, frame_kdtree, max_distance=allowed_max_distance,
                                            detection_cost_fn=self.detection_cost_fn, seconds_passed=seconds_since_last_frame)
        candidates = list(candidates)
        if len(candidates) > 0:
            fill_distance_matrix(candidates, cost_matrix)
            tracklet_indices, detection_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
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

            linked_tracklet_indices.add(tracklet_idx)
            linked_detection_indices.add(tracklet_idx)

        # Uncontinued tracklets are closed.
        self.minimum_open_tracklet_begin = None
        old_open_tracklets = self.open_tracklets
        self.open_tracklets = []
        for idx, tracklet in enumerate(old_open_tracklets):
            if idx in linked_tracklet_indices:
                if self.minimum_open_tracklet_begin is None or (tracklet.timestamps[0] < self.minimum_open_tracklet_begin):
                    self.minimum_open_tracklet_begin = tracklet.timestamps[0]
                self.open_tracklets.append(tracklet)
            else:
                yield tracklet
        
        # Unassigned detections become new tracklets.
        self.push_detections_as_new_tracklets((detection for idx, detection in enumerate(frame_detections) if not (idx in linked_detection_indices)),
                                              frame_id, frame_datetime)