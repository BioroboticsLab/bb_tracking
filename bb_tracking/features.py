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

def detection_angular_distance(*detections, norm=None):
    if norm is None:
        norm = detection_temporal_distance(*detections)
        assert norm > 0
    a0, a1 = detections[0].orientation_hive, detections[1].orientation_hive
    return np.arctan2(np.sin(a1-a0), np.cos(a1-a0)) \
            / norm

def bitwise_distance(bits0, bits1, fun):
    if bits0 is None and bits1 is None:
        return 0.0
    elif bits0 is None or bits1 is None:
        return 1.0
    return fun(np.abs(np.array(bits0) - np.array(bits1)))

def bitwise_manhattan_distance(bits0, bits1):
    return bitwise_distance(bits0, bits1, np.mean)

def detection_id_distance(*detections):
    bits0, bits1 = detections[0].bit_probabilities, detections[1].bit_probabilities
    return bitwise_manhattan_distance(bits0, bits1)
    
def get_detection_features(*detections):
    return (detection_distance(*detections),
            detection_angular_distance(*detections),
            detection_id_distance(*detections))

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