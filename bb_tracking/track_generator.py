import numpy as np
import scipy.optimize
from . import tracklet_generator
from . import types
import bb_utils.ids

def match_track_lists(open_tracks, new_tracklets, tracklet_cost_fn, max_seconds_gap, **kwargs):
    for i, open_track in enumerate(open_tracks):
        for j, tracklet in enumerate(new_tracklets):
            gap_duration_seconds = (tracklet.timestamps[0] - open_track.timestamps[-1]).total_seconds()

            if gap_duration_seconds <= 0.0 or gap_duration_seconds > max_seconds_gap:
                continue
            if not (open_track.detections[-1].timestamp < tracklet.detections[0].timestamp):
                print (gap_duration_seconds)
                print(open_track)
                print(tracklet)
            assert open_track.detections[-1].timestamp < tracklet.detections[0].timestamp

            yield i, j, tracklet_cost_fn(open_track, tracklet)

def assign_tracked_bee_id(track):
    bits = []
    for detection in track.detections:
        if detection.detection_type == types.DetectionType.TaggedBee:
            bits.append(detection.bit_probabilities)
    if len(bits) == 0:
        return None
    bits = np.stack(bits, axis=0)
    bee_id = np.median(bits, axis=0)
    bee_id = bb_utils.ids.BeesbookID.from_bb_binary(bee_id).as_ferwar()
    return bee_id

class TrackGenerator():
    def __init__(self, tracklet_generator, tracklet_cost_fn, max_cost=1.0, max_seconds_gap=5.0):
        self.tracklet_generator = tracklet_generator
        self.tracklet_cost_fn = tracklet_cost_fn

        self.max_cost = max_cost
        self.max_seconds_gap = max_seconds_gap

        self.open_tracks = []

    def finalize_track(self, track):
        tracked_bee_id = assign_tracked_bee_id(track)
        track = track._replace(bee_id = tracked_bee_id)
        return track

    def finalize_all(self):
        closed_tracklets = self.tracklet_generator.finalize_all()
        yield from self.push_tracklets(closed_tracklets)
        yield from (self.finalize_track(t) for t in self.open_tracks)
        self.open_tracks = []

    def push_frame(self, *args):
        closed_tracklets = self.tracklet_generator.push_frame(*args)
        yield from self.push_tracklets(closed_tracklets)

    def push_tracklets(self, tracklets):
        
        tracklets = list(tracklets)

        if len(self.open_tracks) == 0:
            self.open_tracks = tracklets
            return
        
        n_open_tracks = len(self.open_tracks)
        n_new_tracklets = len(tracklets)
        cost_matrix = np.ones(shape=(n_open_tracks, n_new_tracklets), dtype=np.float32) + self.max_cost + 1e10

        candidates = match_track_lists(self.open_tracks, tracklets, self.tracklet_cost_fn, self.max_seconds_gap)
        candidates = list(candidates)

        if len(candidates) > 0:
            tracklet_generator.fill_distance_matrix(candidates, cost_matrix)
            track_indices, tracklet_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
        else:
            track_indices, tracklet_indices = tuple(), tuple()

        linked_track_indices = set()
        linked_tracklet_indices = set()
        for (track_idx, tracklet_idx) in zip(track_indices, tracklet_indices):
            cost = cost_matrix[track_idx, tracklet_idx]
            if cost > self.max_cost:
                continue
            track = self.open_tracks[track_idx]
            track.detections.extend(tracklets[tracklet_idx].detections)
            track.timestamps.extend(tracklets[tracklet_idx].timestamps)
            track.frame_ids.extend(tracklets[tracklet_idx].frame_ids)

            linked_track_indices.add(track_idx)
            linked_tracklet_indices.add(tracklet_idx)
        
        # Possibly close tracks that have been open for too long.
        minimum_open_tracklet_datetime = self.tracklet_generator.get_minimum_open_tracklet_begin()
        if minimum_open_tracklet_datetime is not None:
            old_open_tracks = self.open_tracks
            self.open_tracks = []
            for idx, track in enumerate(old_open_tracks):
                if idx in linked_track_indices:
                    self.open_tracks.append(track)
                else:
                    gap_duration_seconds = (minimum_open_tracklet_datetime - track.timestamps[-1]).total_seconds()
                    if gap_duration_seconds > self.max_seconds_gap:
                        yield self.finalize_track(track)
                    else:
                        self.open_tracks.append(track)

        # Add unmatched tracklets as new open tracks.
        for i, tracklet in enumerate(tracklets):
            if i not in linked_tracklet_indices:
                self.open_tracks.append(tracklet)