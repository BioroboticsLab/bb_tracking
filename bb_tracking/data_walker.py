from collections import defaultdict
import datetime, pytz
import bb_binary
import numpy as np
import scipy.spatial
import bb_utils.ids
from . import types

def get_hive_coordinates(x, y, orientation, H):
    if H is None:
        return x, y, orientation

    xy = np.ones(shape=(3, 2), dtype=np.float32)
    xy[0, 0], xy[1, 0] = x, y
    xy[0, 1], xy[1, 1] = x + np.cos(orientation) * 10.0, y + np.sin(orientation) * 10.0
    xy_mm = np.dot(H, xy).T
    xy_mm[:, :2] /= xy_mm[:, 2:]

    orientation_hive = np.arctan2(xy_mm[1, 1] - xy_mm[0, 1], xy_mm[1, 0] - xy_mm[0, 0])
    
    return xy_mm[0, 0], xy_mm[0, 1], orientation_hive
    
def make_detection(bee, H=None, frame_id=None, timestamp=None, orientation=np.nan, detection_type=types.DetectionType.Invalid,
                   bit_probabilities=None, is_truth=False, no_datetime_timestamps=False):
    x, y = bee.xpos, bee.ypos
    x_hive, y_hive, orientation_hive = get_hive_coordinates(x, y, orientation, H)
    localizerSaliency = np.nan if is_truth else bee.localizerSaliency
    
    has_posix_timestamp = isinstance(timestamp, (float, np.floating))
    if not has_posix_timestamp:
        posix_timestamp = timestamp.timestamp()
    else:
        posix_timestamp = timestamp

    if no_datetime_timestamps:
        timestamp = None
    elif has_posix_timestamp:
        timestamp = datetime.datetime.fromtimestamp(posix_timestamp, tz=pytz.UTC)

    return types.Detection(x, y, orientation, 
                           x_hive, y_hive, orientation_hive,
                           timestamp, posix_timestamp, frame_id,
                           detection_type, bee.idx, localizerSaliency,
                           bit_probabilities)


def iterate_bb_binary_repository(repository_path, dt_begin, dt_end, homography_fn=None,
                                 is_truth_repos=False, only_tagged_bees=False, cam_id=None,
                                 is_2019_repos=True, no_datetime_timestamps=False):
    repo = bb_binary.Repository(repository_path)

    for fc_path in repo.iter_fnames(begin=dt_begin, end=dt_end, cam=cam_id):
        fc = bb_binary.load_frame_container(fc_path)
        cam_id = fc.camId

        for frame in fc.frames:
            frame_id = frame.id
            frame_timestamp = frame.timestamp
            frame_datetime = pytz.UTC.localize(datetime.datetime.utcfromtimestamp(frame_timestamp))
            if (dt_begin is not None) and (frame_datetime < dt_begin):
                continue
            if (dt_end is not None) and (frame_datetime >= dt_end):
                continue

            H = None
            if homography_fn is not None:
                H = homography_fn(cam_id, frame_datetime)

            frame_detections = list()
            if not is_truth_repos:
                # Tagged bees.
                bee_generator = frame.detectionsDP if is_2019_repos else frame.detectionsUnion.detectionsDP
                for bee in bee_generator:
                    frame_detections.append(make_detection(
                        bee, frame_id=frame_id, timestamp=frame_datetime, orientation=bee.zRotation,
                        H=H, detection_type=types.DetectionType.TaggedBee, bit_probabilities=np.array(bee.decodedId) / 255.0,
                        no_datetime_timestamps=no_datetime_timestamps))
            else:
                # GT bees.
                bee_generator = frame.detectionsTruth if is_2019_repos else frame.detectionsUnion.detectionsTruth
                for bee in bee_generator:
                    numeric_id = bee.decodedId
                    bit_probabilities = bb_utils.ids.BeesbookID.from_dec_12_reverse(numeric_id).as_bb_binary()

                    frame_detections.append(make_detection(
                        bee, frame_id=frame_id, timestamp=frame_datetime,
                        H=H, detection_type=types.DetectionType.TaggedBee, bit_probabilities=bit_probabilities,
                        is_truth=True, no_datetime_timestamps=no_datetime_timestamps))

            if not only_tagged_bees:
                # Untagged bees.
                for bee in frame.detectionsBees:
                    detection_type = dict(untagged=types.DetectionType.UntaggedBee,
                                      inCell=types.DetectionType.BeeInCell,
                                      upsideDown=types.DetectionType.BeeOnGlass)[str(bee.type)]
                    frame_detections.append(make_detection(
                        bee, frame_id=frame_id, timestamp=frame_datetime,
                        H=H, detection_type=detection_type, bit_probabilities=None,
                        no_datetime_timestamps=no_datetime_timestamps))
            
            xy = [(detection.x_hive, detection.y_hive) for detection in frame_detections]
            frame_kdtree = scipy.spatial.cKDTree(xy)

            yield (cam_id, frame_id, frame_datetime,
                    frame_detections, frame_kdtree)

