from collections import namedtuple, defaultdict, deque
import datetime, pytz
import numpy as np
import pickle
import scipy.spatial
import bb_utils
import bb_tracking.data.datastructures
from .tracklet_generator import generate_random_track_id
from .data_walker import get_hive_coordinates, iterate_bb_binary_repository
from . import types

Learningtracks = namedtuple('Learningtracks', ('frame_objects_path', 'frame_objects_test', 'learning_target'))


def convert_detection(old_detection : bb_tracking.data.datastructures.Detection, H):
    x, y, orientation = old_detection.x, old_detection.y, old_detection.orientation
    x_hive, y_hive, orientation_hive = get_hive_coordinates(x, y, orientation, H)
    frame_id = int(old_detection.id[1:].split("d")[0])
    timestamp = pytz.UTC.localize(datetime.datetime.utcfromtimestamp(old_detection.timestamp))
    return types.Detection(x, y, orientation,
                           x_hive, y_hive, orientation_hive,
                           timestamp, timestamp.timestamp(),
                           frame_id,
                           types.DetectionType.TaggedBee, 0, old_detection.meta["localizerSaliency"],
                           old_detection.beeId)

def convert_track(old_track : bb_tracking.data.datastructures.Track, H):
    datetimes = [pytz.UTC.localize(datetime.datetime.utcfromtimestamp(ts)) for ts in old_track.timestamps]
    # Assume all tracks belong to the same cam.
    cam_id = old_track.meta["detections"][0].meta["camId"]
    # Fake frame IDs.
    frame_ids = [generate_random_track_id() for _ in range(len(datetimes))]
    detections = [convert_detection(d, H=H) for d in old_track.meta["detections"]]

    return types.Track(generate_random_track_id(), cam_id, detections, datetimes, frame_ids, None, dict())

def load_ground_truth_tracks(path="/mnt/storage/beesbook/learning_data/learning_fragments_framediff17_dataset20150918_Truth.p",
                                N=None,
                                detection_feature_fn=None,
                                track_feature_fn=None):
    """Loads available GT tracks from 2015.
    """


    scale = 3.0 / 50.0
    static_H = np.array([[scale, 0, 0],
                         [0, scale, 0],
                         [0, 0, 1.0]], dtype=np.float32)

    with open(path, "rb") as f:
        fragment_data = pickle.load(f)

    inconsistent_track_count = 0
    for idx, metadata in enumerate(fragment_data):
        track0, track1, target = ((convert_track(metadata.frame_objects_path, H=static_H), 
                           convert_track(metadata.frame_objects_test, H=static_H),
                           metadata.learning_target))
        # Skip tracks that are temporally inconsistent.
        if track0.timestamps[-1] >= track1.timestamps[0]:
            inconsistent_track_count += 1
            continue
        if detection_feature_fn is not None:
            yield detection_feature_fn(track0.detections[-1], track1.detections[0]), target
        elif track_feature_fn is not None:
            yield track_feature_fn(track0, track1), target
        else:
            yield track0, track1, target

        if N is not None and idx == N - 1:
            break
    
    if inconsistent_track_count > 0:
        print("Inconsistent tracks in the data: {}".format(inconsistent_track_count))

def iterate_ground_truth_repository(gt_repo_path, pipeline_repo_path, max_distance=3, homography_fn=None, verbose=True):

    pipeline_data_generators_for_cam_id = dict()

    stats = defaultdict(lambda: 0)

    def advance_pipeline_data(dt_begin, cam_id):
        """Wraps access to bb_binary repository iterators.
        Can be called to retrieve the next frame data for a specific cam ID.
        """
        nonlocal pipeline_data_generators_for_cam_id

        if cam_id not in pipeline_data_generators_for_cam_id:
            pipeline_data_generators_for_cam_id[cam_id] = iterate_bb_binary_repository(pipeline_repo_path, dt_begin=dt_begin, dt_end=None,
                                                                           homography_fn=homography_fn, cam_id=cam_id)
        return next(pipeline_data_generators_for_cam_id[cam_id])

    # Iterate over ground truth data.
    for (cam_id, frame_id, frame_datetime, frame_detections, _) in iterate_bb_binary_repository(
            gt_repo_path, None, None, homography_fn=homography_fn, is_truth_repos=True, only_tagged_bees=True):
        matching_frame_found = False
        pipeline_data = None
        # Look for matching pipeline data based on timestamp.
        while not matching_frame_found:
            pipeline_data = advance_pipeline_data(frame_datetime, cam_id)
            assert pipeline_data[0] == cam_id

            if pipeline_data[2] < frame_datetime:
                continue
            elif pipeline_data[2] > frame_datetime:
                raise Exception("Matching frame not found!")
                break
            matching_frame_found = True
            break

        assert matching_frame_found
        assert pipeline_data[0] == cam_id

        # Match GT to frame.
        used_pipeline_indices = set() # Which detection indices have already been used? Don't assign twice.
        matched_detections = []
        _, pipeline_frame_id, _, pipeline_detections, pipeline_kdtree = pipeline_data
        for gt_detection in frame_detections:
            candidate, use_gt_detection = None, False
            candidates = pipeline_kdtree.query_ball_point((gt_detection.x_hive, gt_detection.y_hive), max_distance)

            if len(candidates) == 0:
                """print("No candidate found :(")
                candidates = pipeline_kdtree.query_ball_point((gt_detection.x_hive, gt_detection.y_hive), max_distance * 10)
                if len(candidates) > 0:
                    xy = []
                    for i in candidates:
                        xy.append((pipeline_detections[i].x_hive, pipeline_detections[i].y_hive))
                    xy = np.array(xy)
                    distances = np.linalg.norm(xy - np.array([gt_detection.x_hive, gt_detection.y_hive]), axis=1)
                    min_idx = np.argmin(distances)
                    print("Closest at {}".format(list(distances)))"""
                # Just use the GT data..
                candidate = gt_detection._replace(detection_index=-gt_detection.detection_index, frame_id=pipeline_frame_id)
                use_gt_detection = True
            elif len(candidates) == 1:
                candidate = pipeline_detections[candidates[0]]
            else:
                xy = []
                for i in candidates:
                    xy.append((pipeline_detections[i].x_hive, pipeline_detections[i].y_hive))
                xy = np.array(xy)
                distances = np.sum(np.abs(xy - np.array([gt_detection.x_hive, gt_detection.y_hive])), axis=1)
                min_idx = np.argmin(distances)
                candidate = pipeline_detections[candidates[min_idx]]
                #print("Multiple candidates at {}".format(list(distances)))

            assert candidate is not None

            if not use_gt_detection:
                if candidate.detection_index in used_pipeline_indices:
                    print("Warning! Would assign pipeline detection to two GT detections. {}|{} vs {}|{}".format(
                            candidate.x_pixels, candidate.y_pixels,
                            gt_detection.x_pixels, gt_detection.y_pixels))
                    print("\t{},{} vs {},{} = {}".format(
                            candidate.x_hive, candidate.y_hive,
                            gt_detection.x_hive, gt_detection.y_hive,
                            np.linalg.norm(np.array([candidate.x_hive, candidate.y_hive]) - np.array([gt_detection.x_hive, gt_detection.y_hive]))))
                used_pipeline_indices.add(candidate.detection_index)

                if verbose:
                    gt_id = bb_utils.ids.BeesbookID.from_bb_binary(gt_detection.bit_probabilities).as_ferwar()
                    pipeline_id = bb_utils.ids.BeesbookID.from_bb_binary(candidate.bit_probabilities).as_ferwar()
                    if gt_id == pipeline_id:
                        stats["ID_match"] += 1
                    else:
                        stats["ID_mismatch"] += 1
            else: # Using gt detection:
                if verbose:
                    stats["detection_false_negatives"] += 1

            matched_detections.append(candidate._replace(x_pixels=gt_detection.x_pixels,
                                                            y_pixels=gt_detection.y_pixels,
                                                            x_hive=gt_detection.x_hive,
                                                            y_hive=gt_detection.y_hive,
                                                            detection_index=gt_detection.detection_index,
                                                            bit_probabilities=gt_detection.bit_probabilities))
        try:
            xy = [(detection.x_hive, detection.y_hive) for detection in matched_detections]
            matched_kdtree = scipy.spatial.cKDTree(xy)
        except:
            print(xy, len(frame_detections), len(matched_detections), len(pipeline_detections))
            print(frame_datetime, pipeline_data[2])
            print(frame_id, pipeline_data[1], pipeline_data[0], cam_id)
            print(frame_detections, pipeline_detections)
            raise

        if verbose:
            assert len(used_pipeline_indices) <= len(pipeline_detections)
            stats["detection_false_positives"] += len(pipeline_detections) - len(used_pipeline_indices)
            stats["detection_true_positives"] += len(used_pipeline_indices)
            stats["n_frames"] += 1

        yield (cam_id, pipeline_frame_id, frame_datetime, matched_detections, matched_kdtree)

    if verbose:
        stats["ID_mismatch_rate"] = stats["ID_mismatch"] / (stats["ID_match"] + stats["ID_mismatch"])
        stats["detection_precision"] = stats["detection_true_positives"] / (stats["detection_false_positives"] + stats["detection_true_positives"])
        stats["detection_recall"] = stats["detection_true_positives"] / (stats["detection_false_negatives"] + stats["detection_true_positives"])
        
        for key, value in stats.items():
            print("{:30s}:\t{:3.3f}".format(key, value))