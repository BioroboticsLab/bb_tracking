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
from .. import types
from .. import features
from .. import models
from .. import evaluate_tracks
from .. import repository_tracker
from . import detection_data_generation

def get_all_tracklets(ground_truth_repos_path, detection_model_path, homography_fn, cam_ids=(0,)):

    # import joblib
    # with open(detection_model_path, "rb") as f:
    #         detection_model = joblib.load(f)

    import xgboost as xgb

    # Load the detection model
    detection_model_booster = xgb.Booster()
    detection_model_booster.load_model(detection_model_path)
    # Wrap the Booster in an XGBClassifier
    detection_model = XGBClassifier()
    detection_model._Booster = tracklet_model_booster




    detection_classification_threshold = 0.6

    detection_kwargs = dict(
        max_distance_per_second = 30.0,
        n_features=18,
        detection_feature_fn=features.get_detection_features,
        detection_cost_fn=lambda f: 1.0 - detection_model.predict_proba(f)[:, 1],
        max_cost=1.0 - detection_classification_threshold
        )

    # Don't compose tracks at all.
    tracklet_kwargs = dict(
        max_distance_per_second = 0.0,
        max_seconds_gap=0.0,
        n_features=1,
        tracklet_feature_fn=lambda *x: (0,),
        tracklet_cost_fn=lambda *x: 1.0,
        max_cost=0.0
        )

    pipeline_tracker = repository_tracker.RepositoryTracker(ground_truth_repos_path,
                                                                    dt_begin=None, dt_end=None,
                                                                      cam_ids=cam_ids,
                                                                      homography_fn=homography_fn,
                                                                      tracklet_kwargs=detection_kwargs,
                                                                      track_kwargs=tracklet_kwargs,
                                                                      repo_kwargs=dict(only_tagged_bees=False))

    return list(pipeline_tracker)

def get_tracklet_features(all_tracklets, get_all_gt_tracks_for_track, timestamp_index,
                         max_gap_size=24, max_speed_per_second=20.0):
    import itertools, collections
    all_timestamps = sorted(set(itertools.chain(*[t.timestamps for t in all_tracklets])))
    timestamp_to_index = {t: i for i, t in enumerate(all_timestamps)}

    def get_likely_gt_track_id_for_tracklet(tracklet):
        all_gt_track_ids = [t.id for t in get_all_gt_tracks_for_track(tracklet)]
        if len(all_gt_track_ids) == 0:
            return None
        most_common = max(all_gt_track_ids, key=all_gt_track_ids.count)
        return most_common

    tracks_by_start_index = collections.defaultdict(list)
    tracks_by_end_index = collections.defaultdict(list)
    for t in all_tracklets:
        tracks_by_start_index[timestamp_to_index[t.timestamps[0]]].append(t)
        tracks_by_end_index[timestamp_to_index[t.timestamps[-1]]].append(t)

    track_results = []
    for right_tracklet in tracks_by_start_index[timestamp_index]:
        right_tracklet_gt_id = get_likely_gt_track_id_for_tracklet(right_tracklet)
        for left_end_timestamp in range(max(0, timestamp_index - max_gap_size - 1), timestamp_index):
            for left_tracklet in tracks_by_end_index[left_end_timestamp]:

                left_detection = left_tracklet.detections[-1]
                right_detection = right_tracklet.detections[0]
                valid_positive_sample = detection_data_generation.is_valid_detection_pair_combination(left_detection, right_detection)

                gap_length_n_frames = (timestamp_to_index[right_detection.timestamp] - timestamp_to_index[left_detection.timestamp] - 1)
                assert gap_length_n_frames <= max_gap_size

                if max_speed_per_second is not None:
                    necessary_distance_per_second = features.detection_distance(left_detection, right_detection)
                    if necessary_distance_per_second > max_speed_per_second:
                        continue

                left_tracklet_gt_id = get_likely_gt_track_id_for_tracklet(left_tracklet)
                tracklet_pair_features = features.get_track_features(left_tracklet, right_tracklet)
                target = int((left_tracklet_gt_id == right_tracklet_gt_id) \
                             and (left_tracklet_gt_id is not None) \
                             and (right_tracklet_gt_id is not None)
                             and valid_positive_sample)

                meta = [timestamp_to_index[left_detection.timestamp], left_detection.x_pixels, left_detection.y_pixels,
                        timestamp_to_index[right_detection.timestamp], right_detection.x_pixels, right_detection.y_pixels,
                        left_tracklet.bee_id, right_tracklet.bee_id,
                        gap_length_n_frames, len(left_tracklet.detections), len(right_tracklet.detections)]

                track_results.append((tracklet_pair_features, target, meta))


    return track_results

def calculate_all_features(ground_truth_repos_path, all_ground_truth_tracks, detection_model_path, homography_fn, n_frames, ground_truth_cam_ids=(0,)):
    all_tracklets = get_all_tracklets(ground_truth_repos_path, detection_model_path, homography_fn=homography_fn, cam_ids=ground_truth_cam_ids)

    gt_track_id_to_track, gt_detection_to_track_id, gt_detection_to_next_detection, get_all_gt_tracks_for_track = \
        evaluate_tracks.prepare_ground_truth_track_mapping(all_ground_truth_tracks, progress_bar=lambda x,**y: x, sanity_check=True)

    all_features = []
    for i in range(1, n_frames):
        features_XYM = get_tracklet_features(all_tracklets, get_all_gt_tracks_for_track, i)
        all_features.append(features_XYM)

    return all_features
