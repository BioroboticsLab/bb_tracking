import bisect
from collections import defaultdict
import concurrent.futures
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import scipy.spatial
import tqdm.auto
from .. import types
from .. import features


def generate_features_for_timestamp(timestamp, current_tracks, candidate_tracks_tree, all_timestamps, timestamp_to_index,
                                    max_speed_per_second=20.0, max_gap_length_n_frames=30):
        
    def cut_track(track, cut_timestamp, take_left_part=True):
        if not (type(track) is types.Track):
            display(track)
        assert type(track) is types.Track
        ts_index = bisect.bisect_left(track.timestamps, cut_timestamp)

        if take_left_part:
            new_track = track._replace(detections=track.detections[:ts_index],
                                       timestamps=track.timestamps[:ts_index],
                                       frame_ids=track.frame_ids[:ts_index])
        else:
            new_track = track._replace(detections=track.detections[ts_index:],
                                       timestamps=track.timestamps[ts_index:],
                                       frame_ids=track.frame_ids[ts_index:])
        return new_track
    
    candidate_kd_tree, all_candidate_tracks = candidate_tracks_tree
    true_positive_gap_sizes = []
    track_results = []
    for track in current_tracks:
        assert type(track) is types.Track
        right_track = cut_track(track, timestamp, take_left_part=False)
        right_track_first_timestamp_index = timestamp_to_index[right_track.timestamps[0]]
         
        if len(right_track.detections) == 0:
            continue

        assert right_track.timestamps[0] >= timestamp
        
        det = right_track.detections[0]
        det_xy = np.array([det.x_hive, det.y_hive])
        # Take the closest candidates.
        _, candidate_indices = candidate_kd_tree.query(det_xy, 20)
        # And throw in a bunch of random ones. Unlikely candidates will be filtered out anyway.
        candidate_indices = set(candidate_indices) | set(np.random.choice(len(all_candidate_tracks), 100))
        candidate_tracks = [all_candidate_tracks[i] for i in candidate_indices]

        if track not in candidate_tracks:
            candidate_tracks.append(track)
        
        for candidate in candidate_tracks:
            for gap_samples in range(3):
                assert timestamp >= candidate.timestamps[0]
                assert type(candidate) is types.Track
                target = int(candidate.id == right_track.id)
                last_candidate_cut_timestamp_index = min(right_track_first_timestamp_index - 1, timestamp_to_index[candidate.timestamps[-1]])

                earliest_allowed_cut_index = right_track_first_timestamp_index - 1 - max_gap_length_n_frames
                max_gap = last_candidate_cut_timestamp_index - earliest_allowed_cut_index
                if max_gap <= 0:
                    continue
                has_enough_true_positive_gap_samples = len(true_positive_gap_sizes) > 10

                assert earliest_allowed_cut_index > 0
                assert max_gap > 0

                if gap_samples > 0 and max_gap <= 2:
                    continue
                
                if gap_samples >= 0:
                    if target == 1:
                        sample_gap = max_gap
                        gap = np.random.randint(0, sample_gap)
                    elif has_enough_true_positive_gap_samples:
                        p = max_gap_length_n_frames / (np.array(true_positive_gap_sizes) + 1)
                        p /= p.sum()
                        gap = min(np.random.choice(true_positive_gap_sizes, p=p), max_gap)
                        assert gap >= 0
                    else:
                        continue
                    candidate_cut_timestamp = all_timestamps[last_candidate_cut_timestamp_index - gap + 1]
                    
                assert last_candidate_cut_timestamp_index < right_track_first_timestamp_index
                if candidate_cut_timestamp > timestamp:
                    continue
                left_track = cut_track(candidate, candidate_cut_timestamp, take_left_part=True)
                if len(left_track.detections) == 0:
                    continue

                assert left_track.timestamps[-1] < timestamp
                assert left_track.timestamps[-1] < right_track.timestamps[0]
                
                left_track_last_timestamp_index = timestamp_to_index[left_track.timestamps[-1]]
                gap_length_n_frames = (right_track_first_timestamp_index - left_track_last_timestamp_index) - 1
                assert gap_length_n_frames >= 0
                if gap_length_n_frames > max_gap_length_n_frames:
                    continue
                
                tracklet_pair_features = features.get_track_features(left_track, right_track)
                
                left_detection = left_track.detections[-1]
                right_detection = right_track.detections[0]

                assert (timestamp_to_index[right_detection.timestamp] - timestamp_to_index[left_detection.timestamp] - 1) == gap_length_n_frames

                if max_speed_per_second is not None:
                    necessary_distance_per_second = features.detection_distance(left_detection, right_detection)
                    if necessary_distance_per_second > max_speed_per_second:
                        continue
                if target == 1:
                    true_positive_gap_sizes.append(gap_length_n_frames)
                assert right_detection.timestamp >= timestamp
                assert left_detection.timestamp < timestamp
                assert left_detection.timestamp != right_detection.timestamp
                meta = [timestamp_to_index[left_detection.timestamp], left_detection.x_pixels, left_detection.y_pixels,
                        timestamp_to_index[right_detection.timestamp], right_detection.x_pixels, right_detection.y_pixels,
                        left_track.bee_id, right_track.bee_id,
                        gap_length_n_frames]
                
                track_results.append((tracklet_pair_features, target, meta))
            
    return track_results

def generate_tracklet_features(gt_tracks, verbose=True, FPS=6.0,
    progress_bar=tqdm.auto.tqdm, just_yield_job_arguments=False, skip_n_frames=0, limit_n_frames=None):
    
    all_timestamps = list(sorted(set(itertools.chain(*[t.timestamps for t in gt_tracks]))))
    timestamp_to_index = {t: i for i, t in enumerate(all_timestamps)}
    timestamps_df = pandas.Series(all_timestamps)
    delta = [d.total_seconds() for d in timestamps_df.diff() if not pandas.isnull(d)]
        
    mean_frame_delta = np.mean(delta)
    real_FPS = 1.0/mean_frame_delta
    n_frames = len(all_timestamps)
    if verbose:
        sns.distplot(delta, hist_kws=dict(log=True), kde=False)
        plt.show()
        print("Mean time delta is {} implying {} FPS".format(mean_frame_delta, real_FPS))
        print("Have {} frames, spanning {}".format(n_frames, all_timestamps[-1] - all_timestamps[0]))
    
    timestamp_to_track_id_map = defaultdict(set)
    track_id_to_track_map = dict()

    for track in gt_tracks:
        assert track.id not in track_id_to_track_map
        track_id_to_track_map[track.id] = track
        min_ts, max_ts = timestamp_to_index[track.timestamps[0]], timestamp_to_index[track.timestamps[-1]]
        for ts in all_timestamps[min_ts:(max_ts+1)]:
            timestamp_to_track_id_map[ts].add(track.id)
    
    max_gap_length_timedelta = datetime.timedelta(seconds=5.0)
    max_gap_length_n_frames = int(FPS * max_gap_length_timedelta.total_seconds())

    def get_all_tracks_for_timestamp(timestamp, track_id_to_track_map, timestamp_to_track_id_map):
        tracks = [track_id_to_track_map[ID] for ID in timestamp_to_track_id_map[timestamp]]
        for track in tracks:
            assert timestamp <= max(track.timestamps)
            assert timestamp >= min(track.timestamps)
        return tracks
    
    timestamp_to_detection_trees_map = dict()
    
    for timestamp in all_timestamps:
        tracks = get_all_tracks_for_timestamp(timestamp, track_id_to_track_map, timestamp_to_track_id_map)
        tree_data = []
        for track in tracks:
            detection_idx = bisect.bisect_left(track.timestamps, timestamp)
            det = track.detections[detection_idx]
            tree_data.append([det.x_hive, det.y_hive])
        kd_tree = scipy.spatial.cKDTree(np.array(tree_data))
        
        timestamp_to_detection_trees_map[timestamp] = \
                        (kd_tree, tracks)
    
    n_frames_processed = 0
    max_workers = 8 if not just_yield_job_arguments else 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_results = []
        for timestamp_index, timestamp in enumerate(progress_bar(all_timestamps, desc="Submitting jobs...")):
            if timestamp_index <= skip_n_frames: # Always skip frame 0.
                continue
                
            all_current_tracks = get_all_tracks_for_timestamp(timestamp, track_id_to_track_map,
                                                              timestamp_to_track_id_map)

            for candidate_timestamp_index in range(max(0, timestamp_index - 1), timestamp_index):
                                
                candidate_timestamp = all_timestamps[candidate_timestamp_index]
                assert candidate_timestamp <= timestamp

                candidate_tracks_tree = timestamp_to_detection_trees_map[candidate_timestamp]
                tree, candidate_tracks = candidate_tracks_tree
                
                for candidate in candidate_tracks:
                    assert timestamp >= candidate.timestamps[0]
                    assert candidate_timestamp >= candidate.timestamps[0]
                    assert candidate_timestamp <= candidate.timestamps[-1]
                
                if not just_yield_job_arguments:
                    f = executor.submit(generate_features_for_timestamp, timestamp,
                                        all_current_tracks, candidate_tracks_tree, all_timestamps, timestamp_to_index, max_gap_length_n_frames=max_gap_length_n_frames)
                    future_results.append(f)
                else:
                    yield dict(timestamp=timestamp, current_tracks=all_current_tracks, all_timestamps=all_timestamps,
                                candidate_tracks_tree=candidate_tracks_tree, timestamp_to_index=timestamp_to_index, max_gap_length_n_frames=max_gap_length_n_frames)

            n_frames_processed += 1
            if limit_n_frames is not None and n_frames_processed >= limit_n_frames:
                return
        if just_yield_job_arguments:
            return

        results = []
        for r in progress_bar(future_results, desc="Retrieving results"):
            results += r.result()

    n_positives = sum(f[1] for f in results)
    if verbose:
        print("{} samples: positives: {}, negatives: {}".format(len(results), n_positives, len(results) - n_positives))

    return results