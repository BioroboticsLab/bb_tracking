import collections
import enum

class DetectionType(enum.Enum):
    Invalid = 0
    TaggedBee = 1
    UntaggedBee = 2
    BeeOnGlass = 3
    BeeInCell = 4

Detection = collections.namedtuple("Detection", 
                ["x_pixels", "y_pixels", "orientation_pixels", # Original image pixels.
                 "x_hive", "y_hive", "orientation_hive", # Usually in mm.
                 "timestamp", # pytz.UTC localized datetime.datetime object.
                              # Will be set after the tracking to make use of numba.
                 "timestamp_posix", # Timestamp in seconds since epoch.
                 "frame_id", "detection_type", "detection_index", "localizer_saliency", # Shared by all types of detections.
                 "bit_probabilities" # Unique to tagged bees.
                ])

Track = collections.namedtuple("Track",
                ["id", # Random, 64bit track ID.
                 "cam_id", # Camera ID.
                 "detections", # List of Detection objects.
                 "timestamps", # List of UTC datetime.datetime objects.
                 "frame_ids", # List of frame IDs, matching the timestamps and detections.
                 "bee_id",
                 "cache_", # Dictionary with helpful tracking caches.
                ])

