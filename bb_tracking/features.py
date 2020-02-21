import datetime
import math
import numpy as np
import numba
from . import types
import scipy.spatial.distance

def detection_temporal_distance(detection0, detection1):
    return detection1.timestamp_posix - detection0.timestamp_posix

@numba.njit
def euclidean_distance(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)

def detection_distance(detection0, detection1, norm=np.nan):
    if np.isnan(norm):
        norm = detection_temporal_distance(detection0, detection1)
    return euclidean_distance(detection0.x_hive, detection0.y_hive, detection1.x_hive, detection1.y_hive) \
            / norm

def detection_raw_distance_vectorized(detections_left, detections_right, norm=None):
    coordinates_left = np.nan * np.zeros(shape=(len(detections_left), 3))
    for idx, detection in enumerate(detections_left):
        coordinates_left[idx] = detection.x_hive, detection.y_hive, detection.timestamp.timestamp()

    coordinates_right = np.nan * np.zeros(shape=(len(detections_right), 3))
    for idx, detection in enumerate(detections_right):
        coordinates_right[idx] = detection.x_hive, detection.y_hive, detection.timestamp.timestamp()
    
    distances = scipy.spatial.distance.cdist(coordinates_left[:, :2], coordinates_right[:, :2])
    temporal_distances = -1.0 * scipy.spatial.distance.cdist(coordinates_left[:, 2:3], coordinates_right[:, 2:3], metric=np.subtract)

    return distances, temporal_distances

@numba.njit
def calculate_forward_motion(x0, y0, a0, x1, y1, a1, norm):
    motion_direction = np.array([x1 - x0,
                                 y1 - y0]) / norm
    a0 = np.array([np.sin(a0), np.cos(a0)])
    a1 = np.array([np.sin(a1), np.cos(a1)])
    
    d0, d1 = np.dot(a0, motion_direction), np.dot(a1, motion_direction)
    if np.isnan(d0) and np.isnan(d1):
        return np.nan
    lowest_d = np.nanmin((d0, d1))
    return lowest_d

def detection_forward_motion(*detections, norm=None):
    if norm is None:
        norm = detection_temporal_distance(*detections)
    return calculate_forward_motion(detections[0].x_hive, detections[0].y_hive, detections[0].orientation_hive,
                                    detections[1].x_hive, detections[1].y_hive, detections[1].orientation_hive,
                                    norm)

@numba.njit
def angular_distance(a0, a1, norm):
    diff = abs((a0 - a1) % (2.0 * math.pi))
    if diff >= math.pi:
        diff -= 2.0 * math.pi
    return abs(diff) / norm

def detection_angular_distance(*detections, norm=None):
    if norm is None:
        norm = detection_temporal_distance(*detections)
    a0, a1 = detections[0].orientation_hive, detections[1].orientation_hive
    return angular_distance(a0, a1, norm)

def bitwise_distance(bits0, bits1, fun):
    if bits0 is None and bits1 is None:
        return np.nan
    elif bits0 is None or bits1 is None:
        return np.nan
    return fun(np.abs(np.array(bits0) - np.array(bits1)))

def bitwise_manhattan_distance(bits0, bits1):
    return bitwise_distance(bits0, bits1, np.mean)

def detection_id_distance(*detections):
    bits0, bits1 = detections[0].bit_probabilities, detections[1].bit_probabilities
    return bitwise_manhattan_distance(bits0, bits1)

def detection_type_to_index(t):
    if t == types.DetectionType.TaggedBee:
        return 0
    if t == types.DetectionType.UntaggedBee:
        return 1
    if t == types.DetectionType.BeeOnGlass:
        return 2
    if t == types.DetectionType.BeeInCell:
        return 3
    assert False

def detection_type_changes_mask(*detections):
    values = [0, 0, 0, 0, 0, 0, 0, 0]
    values[detection_type_to_index(detections[0].detection_type)] = 1
    values[4 + detection_type_to_index(detections[1].detection_type)] = 1
    return tuple(values)

def get_detection_confidence(detection):
    if detection.bit_probabilities is None:
        return np.nan
    return 2.0 * np.mean(np.abs(detection.bit_probabilities - 0.5))

def detection_confidences(*detections):
    return (get_detection_confidence(detections[0]),
            get_detection_confidence(detections[1]))

def detection_localizer_saliencies(*detections):
    return (detections[0].localizer_saliency,
            detections[1].localizer_saliency)

def detection_localizer_saliencies_difference(*detections):
    return 2.0 - abs(detections[0].localizer_saliency - detections[1].localizer_saliency)

def get_detection_features(*detections):
    return (detection_distance(detections[0], detections[1]),
            detection_angular_distance(*detections),
            detection_id_distance(*detections),
            detection_localizer_saliencies_difference(*detections),
            detection_forward_motion(*detections),
            ) + \
                detection_confidences(*detections) + \
                detection_localizer_saliencies(*detections) + \
                detection_type_changes_mask(*detections)

def detection_id_match(*detections):
    bits0, bits1 = detections[0].bit_probabilities, detections[1].bit_probabilities
    if bits0 is None and bits1 is None:
        return 1.0
    elif bits0 is None or bits1 is None:
        return 0.0
    
    if ((bits0 > 0.5) ^ (bits1 > 0.5)).sum() > 0:
        return 0.0
    return 1.0

    max_bit_distance = bitwise_distance(detections[0].bit_probabilities, detections[1].bit_probabilities, np.max)
    if max_bit_distance < 0.5:
        return 1.0
    return 0.0

def detection_id_match_cost(*detections):
    return 1.0 - detection_id_match(*detections)

def get_detection_features_id_only(*detections):
    return (detection_id_match(*detections),)

def track_mean_id(tracklet):
    if "track_mean_id" in tracklet.cache_:
        return tracklet.cache_["track_mean_id"]

    tracklet_bits = np.array([d.bit_probabilities for d in tracklet.detections if d.detection_type == types.DetectionType.TaggedBee])
    if tracklet_bits.shape[0] == 0:
        mean_id = None
    else:
        mean_id = np.mean(tracklet_bits, axis=0)

    tracklet.cache_["track_mean_id"] = mean_id
    return mean_id

def track_id_distance(*tracklets):
    return bitwise_manhattan_distance(track_mean_id(tracklets[0]), track_mean_id(tracklets[1]))

def track_distance(*tracklets):
    return detection_distance(tracklets[0].detections[-1], tracklets[1].detections[0])

@numba.njit
def short_angle_dist(a0,a1):
    """Returns the signed distance between two angles in radians.
    """
    max = np.pi*2
    da = (a1 - a0) % max
    return 2*da % max - da

@numba.njit
def extrapolate_position_and_angles(x0, y0, a0, x1, y1, a1, factor):
    x = x1 + (x1 - x0) * factor
    y = y1 + (y1 - y0) * factor
    a = a1 + short_angle_dist(a0, a1) * factor
    return x, y, a


def extrapolate_detections(seconds, *detections):
    if len(detections) == 1:
        return detections[0]

    seconds_distance = detection_temporal_distance(*detections)
    extrapolation_factor = seconds / seconds_distance

    x, y, orientation = extrapolate_position_and_angles(detections[0].x_hive, detections[0].y_hive, detections[0].orientation_hive,
                                             detections[1].x_hive, detections[1].y_hive, detections[1].orientation_hive,
                                             extrapolation_factor)

    return types.Detection(None, None, None,
            x, y, orientation,
            None, detections[1].timestamp_posix + seconds, 0,
            detections[1].detection_type, None, None, None)

def track_forward_distance(*tracklets):
    seconds_distance = detection_temporal_distance(tracklets[0].detections[-1], tracklets[1].detections[0])
    return detection_distance(extrapolate_detections(seconds_distance, *tracklets[0].detections[-2:]), tracklets[1].detections[0], norm=1.0)

def track_backward_distance(*tracklets):
    seconds_distance = abs(detection_temporal_distance(tracklets[0].detections[-1], tracklets[1].detections[0]))
    return detection_distance(extrapolate_detections(seconds_distance, *tracklets[1].detections[1::-1]), tracklets[0].detections[-1], norm=1.0)

def track_angular_distance(*tracklets):
    return detection_angular_distance(tracklets[0].detections[-1], tracklets[1].detections[0])

def track_decoder_confidence(tracklet):
    if "track_decoder_confidence" in tracklet.cache_:
        return tracklet.cache_["track_decoder_confidence"]

    bits = [d.bit_probabilities for d in tracklet.detections if d.detection_type == types.DetectionType.TaggedBee]
    if len(bits) != 0:
        bits = np.median(np.array(bits), axis=0)
        confidence = np.min(np.abs(bits - 0.5))
    else:
        confidence = 0.0
    tracklet.cache_["track_decoder_confidence"] = confidence
    return confidence

def track_difference_of_confidence(*tracklets):
    return abs(track_decoder_confidence(tracklets[0]) - track_decoder_confidence(tracklets[1]))

def get_track_features(*tracks):
    return (track_id_distance(*tracks),
            track_distance(*tracks),
            track_forward_distance(*tracks),
            track_backward_distance(*tracks),
            track_angular_distance(*tracks),
            track_difference_of_confidence(*tracks))

def track_id_match_cost(*tracks):
    return detection_id_match_cost(tracks[0].detections[-1], tracks[1].detections[0])

def get_track_features_id_only(*tracks):
    return (track_id_match(),)