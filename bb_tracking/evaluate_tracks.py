from collections import defaultdict
import numpy as np
import pandas
import tqdm.auto

def get_metrics_for_track(track, gt_track, track_false_positives):
    def series_for_track(t, is_false_positive):
        track_frames = []
        track_detection_keys = {d.timestamp: (d.detection_type, d.detection_index) for i, d in enumerate(t.detections) if not is_false_positive(i, d)}
        for key in t.timestamps:
            det_key = track_detection_keys[key] if key in track_detection_keys else (np.nan, np.nan)
            track_frames.append((key, det_key[0], float(det_key[1])))
        track_frames = pandas.DataFrame(track_frames, columns=["datetime", "type_", "idx_"])
        track_frames.set_index("datetime", inplace=True)
        return track_frames
        
    track_series = series_for_track(track, lambda i, d: track_false_positives[i]) # A pipeline detection had no GT match?
    gt_series = series_for_track(gt_track, lambda i, d: d.detection_index < 0) # A GT detection was not available to the pipeline?

    df = track_series.merge(gt_series, how="outer", left_index=True, right_index=True, suffixes=("detection", "gt"))
    #print(df)
    stats = dict()
    stats["n_detections"] = (~pandas.isnull(df.idx_detection)).sum()
    stats["gt_track_len"] = (~pandas.isnull(df.idx_gt)).sum()
    
    valid_idx = np.where(~pandas.isnull(df.idx_detection))[0]
    gap_left, gap_right = 0, 0
    if len(valid_idx) > 0:
        gap_left, gap_right = min(valid_idx), (df.shape[0] - 1) - max(valid_idx)
    stats["gap_left"] = gap_left
    stats["gap_right"] = gap_right
    df = df.iloc[gap_left:(df.shape[0] - gap_right), :].copy()
    stats["track_len"] = df.shape[0]
    assert stats["track_len"] >= stats["n_detections"]
    df["valid_gap"] = pandas.isnull(df.idx_detection) & pandas.isnull(df.idx_gt)
    df["matching_detection"] = (df.idx_detection == df.idx_gt) & (df.type_detection == df.type_gt)
    stats["n_valid_gaps"] = df.valid_gap.sum()
    stats["matches"] = (df.matching_detection | df.valid_gap).sum()
    stats["mismatches"] = ((~df.matching_detection) & (~pandas.isnull(df.idx_detection)) & (~pandas.isnull(df.idx_gt))).sum()
    stats["inserts"] = ((~pandas.isnull(df.idx_detection)) & (pandas.isnull(df.idx_gt))).sum()
    stats["deletes"] = ((pandas.isnull(df.idx_detection)) & (~pandas.isnull(df.idx_gt))).sum()
    stats["n_gaps"] = df.shape[0] - stats["n_detections"]
    stats["correct_track_id"] = int(track.bee_id == gt_track.bee_id)
    stats["correct_detection_id"] = stats["correct_track_id"] * stats["n_detections"]

    track_score = (stats["matches"], stats["n_detections"] / stats["gt_track_len"])

    #print(track.bee_id, gt_track.bee_id)
    #print(stats)

    return track_score, stats



def calculate_metrics_for_tracking(track_generator, ground_truth_track_generator, progress_bar=tqdm.auto.tqdm):

    gt_track_id_to_track = dict()
    gt_detection_to_track_id = dict()

    track_counter = 0
    for gt_track in progress_bar(ground_truth_track_generator, desc="Checking ground truth.."):
        track_counter += 1
        gt_track_id_to_track[gt_track.id] = gt_track
        for detection in gt_track.detections:
            if detection.frame_id not in gt_track.frame_ids:
                print(gt_track)
            assert detection.frame_id in gt_track.frame_ids
            gt_detection_to_track_id[(detection.frame_id, detection.detection_type, detection.detection_index)] = gt_track.id

    for (fid, dtype, didx), track_id in gt_detection_to_track_id.items():
        track = gt_track_id_to_track[track_id]
        fids = [d.frame_id for d in track.detections]
        if fid not in track.frame_ids:
            print(track)
        assert fid in fids
        assert fid in track.frame_ids
        det_keys = [(d.detection_type, d.detection_index) for d in track.detections]
        assert (dtype, didx) in det_keys

    assert len(list(gt_track_id_to_track.keys())) == track_counter
    print("Found {} GT tracks.".format(track_counter))

    def get_all_gt_tracks_for_track(track):
        nonlocal gt_track_id_to_track
        nonlocal gt_detection_to_track_id
        found_track_ids = set()

        for d in track.detections:
            key = (d.frame_id, d.detection_type, d.detection_index)
            if (key in gt_detection_to_track_id) and (key not in found_track_ids):
                found_track_ids.add(key)
                track_id = gt_detection_to_track_id[key]
                matching_gt_track = gt_track_id_to_track[track_id]
                #print("key: {}\tTrack len: {}".format(key, len(track.detections)))
                #print(track)
                #print("GT Track:")
                #print(matching_gt_track)
                yield matching_gt_track

    statistics = defaultdict(float)

    all_track_stats = []
    for track in progress_bar(track_generator, desc="Tracking repository..."):
        false_positives = [(d.frame_id, d.detection_type, d.detection_index) not in gt_detection_to_track_id for d in track.detections]
        max_score, track_stats = None, None
        for gt_track in get_all_gt_tracks_for_track(track):
            score, stats = get_metrics_for_track(track, gt_track, false_positives)
            if max_score is None or score > max_score:
                max_score = score
                track_stats = stats
        if track_stats is not None:
            all_track_stats.append(track_stats)

    return pandas.DataFrame(all_track_stats)