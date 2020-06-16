import bisect
import datetime, pytz
import numpy as np
import hungarian
from . import tracklet_generator
from . import types
from . import features
import bb_utils.ids

def calculate_track_pair_features(open_tracks, new_tracklets, tracklet_feature_fn,
                                    max_distance_per_second, max_seconds_gap, n_features, generate_features=True):
    detections_left = [t.detections[-1] for t in open_tracks]
    detections_right = [t.detections[0] for t in new_tracklets]
    euclidean_distances, temporal_distances = features.detection_raw_distance_vectorized(detections_left, detections_right)
    valid_indices = (temporal_distances > 0) & (temporal_distances < max_seconds_gap)
    if max_distance_per_second is not None:
        euclidean_distances[valid_indices] /= temporal_distances[valid_indices]
        valid_indices = valid_indices & (euclidean_distances <= max_distance_per_second)
    tracklets0_indices, tracklets1_indices = np.where(valid_indices)
    if not generate_features:
        return tracklets0_indices, tracklets1_indices

    all_features = np.nan * np.zeros(shape=(len(tracklets0_indices), n_features), dtype=np.float32)
    for i, (t0, t1) in enumerate(zip(tracklets0_indices, tracklets1_indices)):
        open_track, tracklet = open_tracks[t0], new_tracklets[t1]
        tracklet_pair_features = tracklet_feature_fn(open_track, tracklet)
        all_features[i] = tracklet_pair_features
    return tracklets0_indices, tracklets1_indices, all_features

def assign_tracked_bee_id(track):
    bits = []
    for detection in track.detections:
        if detection.detection_type == types.DetectionType.TaggedBee:
            bits.append(detection.bit_probabilities)
    if len(bits) == 0:
        return None
    bits = np.stack(bits, axis=0)
    bit_confidences = np.abs(bits - 0.5) * 2.0
    confidences = np.mean(np.log1p(bit_confidences), axis=1)
    confidences_idx = np.argsort(confidences)[::1]
    N = max(5, bits.shape[0] // 10)
    good_bits = bits[confidences_idx, :]
    bee_id = np.median(good_bits, axis=0)
    if np.any(np.isnan(bee_id)):
        print("Bee ID calculation failed for track {}".format(track.id))
        print(bits)
        print(bit_confidences)
        print(perc)
        return None
    
    bee_id_confidence = np.prod(np.abs(bee_id - 0.5) * 2.0)
    bee_id = bb_utils.ids.BeesbookID.from_bb_binary(bee_id).as_ferwar()
    return bee_id

class TrackGenerator():
    def __init__(self, tracklet_generator, n_features, tracklet_feature_fn, tracklet_cost_fn,
                    max_distance_per_second=None, max_cost=1.0, max_seconds_gap=5.0, verbose=False):
        self.tracklet_generator = tracklet_generator
        self.n_features = n_features
        self.tracklet_feature_fn = tracklet_feature_fn
        self.tracklet_cost_fn = tracklet_cost_fn
        self.closed_tracklet_queue = [] # New tracklets are first collected and then processed in bulk.

        self.max_cost = max_cost
        self.max_distance_per_second = max_distance_per_second
        self.max_seconds_gap = max_seconds_gap
        self.verbose = verbose

        self.open_tracks = []

    def finalize_track(self, track):
        tracked_bee_id = assign_tracked_bee_id(track)
        track = track._replace(bee_id = tracked_bee_id)

        for idx, detection in enumerate(track.detections):
            if detection.timestamp is None:
                track.detections[idx] = detection._replace(timestamp=datetime.datetime.fromtimestamp(detection.timestamp_posix, tz=pytz.UTC))
        return track

    def finalize_all(self):
        if self.verbose:
            print("Finalizing everything...", flush=True)
        closed_tracklets = self.tracklet_generator.finalize_all()
        self.closed_tracklet_queue += list(closed_tracklets)
        yield from self.process_closed_tracklet_queue(process_all=True)
        yield from (self.finalize_track(t) for t in self.open_tracks)
        self.open_tracks = []

    def process_closed_tracklet_queue(self, process_all=False):
        if self.verbose:
            print("Processing current queue of size {}".format(len(self.closed_tracklet_queue)))
        if len(self.closed_tracklet_queue) == 0:
            return

        min_open_timestamp = self.tracklet_generator.get_first_open_begin_datetime()
        current_timestamp = self.tracklet_generator.get_last_frame_datetime()
        if min_open_timestamp is None:
            min_open_timestamp = current_timestamp
        min_open_timestamp = min(min_open_timestamp, current_timestamp)
        close_tracks_ending_before = min_open_timestamp - datetime.timedelta(seconds=self.max_seconds_gap)

        self.closed_tracklet_queue = sorted(self.closed_tracklet_queue, key=lambda t: t.timestamps[0], reverse=True)
        while True:
            queue = []
            min_closing_end, max_closing_begin = None, None
            min_closing_begin = None
            for idx in range(len(self.closed_tracklet_queue)-1, -1, -1):
                tracklet = self.closed_tracklet_queue[idx]
                if (not process_all) and (tracklet.timestamps[0] >= min_open_timestamp):
                    break
                if min_closing_end is not None:
                    if tracklet.timestamps[0] > min_closing_end:
                        break
                    if tracklet.timestamps[-1] < max_closing_begin:
                        continue
                else:
                    min_closing_begin = tracklet.timestamps[0]
                assert tracklet.timestamps[0] >= min_closing_begin

                queue.append(tracklet)
                del self.closed_tracklet_queue[idx]
                min_closing_end = tracklet.timestamps[-1] if min_closing_end is None else min(min_closing_end, tracklet.timestamps[-1])
                max_closing_begin = tracklet.timestamps[0] if max_closing_begin is None else max(max_closing_begin, tracklet.timestamps[0])


            if len(queue) > 0:
                if self.verbose:
                    print("Processing queue N={:5d} from {} to {}.\n\tat {}".format(len(queue), max_closing_begin, min_closing_end, min_open_timestamp))
                close_before = close_tracks_ending_before
                if process_all:
                    close_before = min_closing_begin - datetime.timedelta(seconds=self.max_seconds_gap)
                yield from self.push_tracklets(queue, close_tracks_ending_before=close_before)
            else:
                break

    def push_frame(self, *args):
        closed_tracklets = self.tracklet_generator.push_frame(*args)
        closed_tracklets = list(closed_tracklets)

        if len(closed_tracklets) == 0:
            if self.verbose:
                print("Next frame. No new tracklets.", flush=True)
            return

        self.closed_tracklet_queue += closed_tracklets
        yield from self.process_closed_tracklet_queue()

    def push_tracklets(self, tracklets, close_tracks_ending_before=None):
        tracklets = list(tracklets)
        if len(tracklets) == 0:
            return
        if len(self.open_tracks) == 0:
            self.open_tracks = tracklets
            return
        n_open_tracks = len(self.open_tracks)
        n_new_tracklets = len(tracklets)

        if self.tracklet_feature_fn is not None:
            tracklet0_indices, tracklet1_indices, all_features = calculate_track_pair_features(
                self.open_tracks, tracklets, self.tracklet_feature_fn, self.max_distance_per_second, self.max_seconds_gap, self.n_features)
            
            square_dimension = max(n_open_tracks, n_new_tracklets)
            cost_matrix = np.zeros(shape=(square_dimension, square_dimension), dtype=np.float32) + 100000.0
        else:
            tracklet0_indices = []

        if len(tracklet0_indices) > 0:
            all_distances = self.tracklet_cost_fn(all_features[:len(tracklet0_indices), :])
            tracklet0_indices = np.array(tracklet0_indices, dtype=np.int32)
            tracklet1_indices = np.array(tracklet1_indices, dtype=np.int32)

            tracklet_generator.fill_distance_matrix(tracklet0_indices, tracklet1_indices, all_distances, cost_matrix)
            lap_results = hungarian.lap(cost_matrix.copy())
            track_indices, tracklet_indices = tuple(range(cost_matrix.shape[0])), lap_results[0]
        else:
            track_indices, tracklet_indices = tuple(), tuple()

        _all_costs = []

        linked_track_indices = set()
        linked_tracklet_indices = set()
        for (track_idx, tracklet_idx) in zip(track_indices, tracklet_indices):
            cost = cost_matrix[track_idx, tracklet_idx]
            if cost > self.max_cost:
                continue
            _all_costs.append(cost)

            track = self.open_tracks[track_idx]
            track.detections.extend(tracklets[tracklet_idx].detections)
            track.timestamps.extend(tracklets[tracklet_idx].timestamps)
            track.frame_ids.extend(tracklets[tracklet_idx].frame_ids)
            track.cache_.clear()

            assert track_idx not in linked_track_indices
            assert tracklet_idx not in linked_tracklet_indices

            linked_track_indices.add(track_idx)
            linked_tracklet_indices.add(tracklet_idx)

        if self.verbose:
            print("\tLinked {}/{} pushed tracklets.".format(len(linked_track_indices), len(tracklets)))
        # Possibly close tracks that have been open for too long.
        closed = 0
        old_open_tracks = self.open_tracks
        self.open_tracks = []

        for idx, track in enumerate(old_open_tracks):
            if idx in linked_track_indices or close_tracks_ending_before is None or track.timestamps[-1] >= close_tracks_ending_before:
                self.open_tracks.append(track)
            else:
                yield self.finalize_track(track)
                closed += 1
                        
        if closed > 0:
            if self.verbose:
                print("Closed {} tracks. Open tracks: {}.".format(closed, len(self.open_tracks)))
        # Add unmatched tracklets as new open tracks.
        for i, tracklet in enumerate(tracklets):
            if i not in linked_tracklet_indices:
                self.open_tracks.append(tracklet)