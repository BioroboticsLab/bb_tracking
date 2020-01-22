import datetime
import numpy as np
import numba
from . import types

def detection_temporal_distance(*detections):
    return (detections[1].timestamp - detections[0].timestamp).total_seconds()

@numba.njit
def euclidean_distance(x0, y0, x1, y1):
    return np.sqrt((x1 - x0) ** 2.0 + (y1 - y0) ** 2.0)

def detection_distance(*detections, norm=None):
    if norm is None:
        norm = detection_temporal_distance(*detections)
        assert norm > 0
    return euclidean_distance(detections[0].x_hive, detections[0].y_hive, detections[1].x_hive, detections[1].y_hive) \
            / norm

def detection_forward_motion(*detections, norm=None):
    if norm is None:
        norm = detection_temporal_distance(*detections)
        assert norm > 0
    motion_direction = np.array([detections[1].x_hive - detections[0].x_hive,
                                 detections[1].y_hive - detections[0].y_hive]) / norm
    a0 = np.array([np.sin(detections[0].orientation_hive), np.cos(detections[0].orientation_hive)])
    a1 = np.array([np.sin(detections[1].orientation_hive), np.cos(detections[1].orientation_hive)])
    
    d0, d1 = np.dot(a0, motion_direction), np.dot(a1, motion_direction)
    if np.isnan(d0) and np.isnan(d1):
        return 0.0
    lowest_d = np.nanmin((d0, d1))
    return lowest_d

def detection_angular_distance(*detections, norm=None):
    if norm is None:
        norm = detection_temporal_distance(*detections)
        assert norm > 0
    a0, a1 = detections[0].orientation_hive, detections[1].orientation_hive
    return np.arctan2(np.sin(a1-a0), np.cos(a1-a0)) \
            / norm

def bitwise_distance(bits0, bits1, fun):
    if bits0 is None and bits1 is None:
        return -1.0
    elif bits0 is None or bits1 is None:
        return 1.0
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
        return -1.0
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
    return (detection_distance(*detections),
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

def track_id_distance(*tracklets):
    bits = []
    for tracklet in tracklets:
        tracklet_bits = np.array([d.bit_probabilities for d in tracklet.detections if d.detection_type == types.DetectionType.TaggedBee])
        if tracklet_bits.shape[0] == 0:
            bits.append(None)
        else:
            bits.append(np.mean(tracklet_bits, axis=0))
    return bitwise_manhattan_distance(*bits)

def track_distance(*tracklets):
    return detection_distance(tracklets[0].detections[-1], tracklets[1].detections[0])

@numba.njit
def short_angle_dist(a0,a1):
    """Returns the signed distance between two angles in radians.
    """
    max = np.pi*2
    da = (a1 - a0) % max
    return 2*da % max - da

def extrapolate_detections(seconds, *detections):
    if len(detections) == 1:
        return detections[0]

    seconds_distance = detection_temporal_distance(*detections)
    timestamp = detections[0].timestamp + datetime.timedelta(seconds=seconds)
    extrapolation_factor = seconds / seconds_distance
    x = detections[1].x_hive + (detections[1].x_hive - detections[0].x_hive) * extrapolation_factor
    y = detections[1].y_hive + (detections[1].y_hive - detections[0].y_hive) * extrapolation_factor
    orientation = detections[1].orientation_hive + short_angle_dist(detections[0].orientation_hive, detections[1].orientation_hive) * extrapolation_factor

    return types.Detection(None, None, None,
            x, y, orientation,
            timestamp, 0,
            detections[0].detection_type, None, None, None)

def track_forward_distance(*tracklets):
    seconds_distance = detection_temporal_distance(tracklets[0].detections[-1], tracklets[1].detections[0])
    return detection_distance(extrapolate_detections(seconds_distance, *tracklets[0].detections[-2:]), tracklets[1].detections[0], norm=1.0)

def track_backward_distance(*tracklets):
    seconds_distance = abs(detection_temporal_distance(tracklets[0].detections[-1], tracklets[1].detections[0]))
    return detection_distance(extrapolate_detections(seconds_distance, *tracklets[1].detections[1::-1]), tracklets[0].detections[-1], norm=1.0)

def track_angular_distance(*tracklets):
    return detection_angular_distance(tracklets[0].detections[-1], tracklets[1].detections[0])

def track_decoder_confidence(tracklet):
    bits = [d.bit_probabilities for d in tracklet.detections if d.detection_type == types.DetectionType.TaggedBee]
    if len(bits) == 0:
        return 0.0
    bits = np.median(np.array(bits), axis=0)
    return np.min(np.abs(bits - 0.5))

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