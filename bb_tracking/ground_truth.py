from . import data_walker
from . import types

from collections import defaultdict
import itertools
import copy
import pickle
import bb_behavior.io
import bb_behavior.utils

def load_ground_truth_tracks(data_path,
                            repository_path):
    """Loads tracks (types.Track) from the results of the ground-truth creation process using the editor UI.
    Takes a .pickle file and a repository and merges the repositories' detections into tracks as merged by the user.
    The bee_id of the tracks will be the user-assigned ID.

    Arguments:
        data_path: string
            Path to the .pickle file that contains the ground-truth as saved by the editor UI.
        repository_path:
            Path to the bb_binary repository that was used in the editor to create the ground truth.
    """
    with open(data_path, "rb") as f:
        ground_truth_data = pickle.load(f)
    
    source_file = ground_truth_data["source"]
    cam_id, dt_begin, dt_end = bb_behavior.io.parse_beesbook_video_filename(source_file)

    result_paths = dict() # path_id -> Track
    path_map = dict() # frame_index, data_source, detection_index -> path_id
    ground_truth_bee_ids = dict() # path_id -> ferwar ID
    frame_indices = set()
    for bee_id, paths in ground_truth_data["paths"].items():
        for _, detections in paths.items():
            random_path_id = bb_behavior.utils.generate_64bit_id()
            assert (random_path_id not in ground_truth_bee_ids) or (ground_truth_bee_ids[random_path_id] == bee_id)
            ground_truth_bee_ids[random_path_id] = bee_id

            frame_indices = set(detections.keys()) | frame_indices
            for frame_index, detection in detections.items():
                detection_index = detection[0]
                data_source = detection[3]
                path_map[(frame_index, data_source, detection_index)] = random_path_id
    
    start_frame_index, end_frame_index = min(frame_indices), max(frame_indices)

    for frame_index, (cam_id, frame_id, frame_datetime, frame_detections, _) in enumerate(
                    data_walker.iterate_bb_binary_repository(repository_path,
                                dt_begin, None, cam_id=cam_id)):
        if frame_index < start_frame_index:
            continue
        if frame_index > end_frame_index:
            break

        n_tagged_bees = sum((1 for d in frame_detections if d.detection_type == types.DetectionType.TaggedBee))

        for detection in frame_detections:
            data_source = 1 if detection.detection_type == types.DetectionType.TaggedBee else 2
            detection_key = (frame_index, data_source, detection.detection_index)

            if detection_key not in path_map:
                continue
            path_id = path_map[detection_key]

            # New path?
            if path_id not in result_paths:
                path = types.Track(path_id, cam_id, [], [], [], ground_truth_bee_ids[path_id])
                result_paths[path_id] = path
            else:
                path = result_paths[path_id]
            
            path.detections.append(detection)
            path.frame_ids.append(frame_id)
            path.timestamps.append(frame_datetime)
    

    return list(result_paths.values())
            

def get_tracks_metadata(tracks):
    timestamp_data = defaultdict(set)
    
    for track in tracks:
        for timestamp in track.timestamps:
            timestamp_data[track.cam_id].add(timestamp)
        #for detection in track.detections:
            
    return {cam_id: list(sorted(v)) for (cam_id, v) in timestamp_data.items()}


def merge_tracks(sets):
    metadatas = [get_tracks_metadata(s) for s in sets]
    assert len(metadatas[0]) == 1 # one cam per set.
    metadatas = [m[list(m.keys())[0]] for m in metadatas]
    
    overlaps = []
    for i in range(len(metadatas)):
        overlaps.append([metadatas[1 - i].index(t) if (t in metadatas[1 - i]) else None \
                   for t in metadatas[i]])
        
    
    if overlaps[0][0] is not None and overlaps[1][-1] is not None:
        metadatas = metadatas[::-1]
        overlaps = overlaps[::-1]
        sets = sets[::-1]
    
    merge_track_map = defaultdict(dict)
    right_track_ids_to_tracks = dict()
    
    for track in sets[1]:
        right_track_ids_to_tracks[track.id] = track
        
        for set0_index, timestamp in zip(overlaps[1], metadatas[1]):
            if set0_index is None:
                break
            if timestamp in track.timestamps:
                detection = track.detections[track.timestamps.index(timestamp)]
                merge_track_map[set0_index][(detection.detection_type,
                                             detection.detection_index)] = track
        
    merged_track_ids = dict() # track_id -> track_id
    
    left_track_ids_to_tracks = dict()
    for track in sets[0]:
        left_track_ids_to_tracks[track.id] = track
        for timestamp, detection in zip(track.timestamps, track.detections):
            frame_index = metadatas[0].index(timestamp)
            overlap_idx = overlaps[0][frame_index]
            if overlap_idx is None:
                continue
            
            detection_key = (detection.detection_type,
                             detection.detection_index)
            if detection_key not in merge_track_map[frame_index]:
                continue
            matching_track_id = merge_track_map[frame_index][detection_key].id
            
            if track.id in merged_track_ids:
                if not (merged_track_ids[track.id] == matching_track_id):
                    print(matching_track_id, merged_track_ids[track.id])
                assert (merged_track_ids[track.id] == matching_track_id)
            
            merged_track_ids[track.id] = matching_track_id
       
    print("Matched {} tracks".format(len(merged_track_ids)))
    
    all_tracks = []
    already_included_ids = set()
    for left_track_id, right_track_id in merged_track_ids.items():
        tracks = left_track_ids_to_tracks[left_track_id], right_track_ids_to_tracks[right_track_id]
        bee_id = tracks[0].bee_id
        if bee_id != tracks[1].bee_id:
            bee_id = [b for b in (tracks[0].bee_id, tracks[1].bee_id) if b is not None][0]
            print("Disagreeing bee IDs {} vs. {} - choosing {}.".format(tracks[0].bee_id, tracks[1].bee_id, bee_id))
        common_frames = [tracks[1].timestamps.index(t) if t in tracks[1].timestamps else None
                         for t in tracks[0].timestamps]
        cutoff_frame_right = min(c for c in common_frames if c is not None)
        cutoff_frame_left = common_frames.index(cutoff_frame_right)
        
        track = tracks[0]
        track = track._replace(
                        bee_id = bee_id,
                        detections = track.detections[:cutoff_frame_left] + tracks[1].detections[cutoff_frame_right:],
                        timestamps = track.timestamps[:cutoff_frame_left] + tracks[1].timestamps[cutoff_frame_right:],
                        frame_ids = track.frame_ids[:cutoff_frame_left] + tracks[1].frame_ids[cutoff_frame_right:]
                        )
        
        all_tracks.append(track)
        already_included_ids.add(track.id)
        already_included_ids.add(tracks[1].id)
        
    for track in itertools.chain(left_track_ids_to_tracks.values(),
                                 right_track_ids_to_tracks.values()):
        if track.id in already_included_ids:
            continue
        all_tracks.append(track)
        
    all_tracks = list(sorted(all_tracks, key=lambda t: t.timestamps[0]))
    return all_tracks

def load_and_merge_ground_truth_tracks(ground_truth_paths, ground_truth_repos_path):
    track_sets = [load_ground_truth_tracks(path, ground_truth_repos_path)
                  for path in ground_truth_paths]
    while len(track_sets) > 1:
        new_set = merge_tracks(track_sets[:2])
        del track_sets[:1]
        track_sets[0] = new_set
    return track_sets[0]