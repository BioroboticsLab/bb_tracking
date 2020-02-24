
import numba
import numpy as np
import prefetch_generator
import tqdm.auto

from . import types
from . import data_walker
from . import ground_truth_legacy

from . import tracklet_generator
from . import track_generator

class CamDataGeneratorTracker():
    def __init__(self, generator, cam_ids, tracklet_kwargs=dict(), track_kwargs=dict(), progress_bar=tqdm.auto.tqdm):
        self.cam_ids = cam_ids
        self.generator = generator
        self.tracklet_kwargs = tracklet_kwargs
        self.track_kwargs = track_kwargs
        self.progress_bar = progress_bar
    
    def __iter__(self):
        trackers = {cam_id: track_generator.TrackGenerator(
                                tracklet_generator.TrackletGenerator(cam_id, **self.tracklet_kwargs),
                                **self.track_kwargs)
                        for cam_id in self.cam_ids}

        untracked_cam_ids = set()
        for (cam_id, frame_id, frame_datetime, frame_detections, frame_kdtree) in \
                self.progress_bar(prefetch_generator.BackgroundGenerator(self.generator(), max_prefetch=5)):
                if cam_id not in trackers:
                    untracked_cam_ids.add(cam_id)
                    continue
                tracker = trackers[cam_id]
                finalized_tracks = tracker.push_frame(frame_id, frame_datetime, frame_detections, frame_kdtree)

                yield from finalized_tracks
        
        if len(untracked_cam_ids) > 0:
            print("Found cam IDs that were not tracked: {}".format(untracked_cam_ids))

        for cam_id, tracker in trackers.items():
            yield from tracker.finalize_all()

class RepositoryTracker(CamDataGeneratorTracker):
    def __init__(self, repo_path,
                 dt_begin, dt_end, homography_fn,
                 **kwargs):

        def yield_from_repository():
            yield from data_walker.iterate_bb_binary_repository(repo_path, dt_begin, dt_end, homography_fn, no_datetime_timestamps=True)

        super().__init__(yield_from_repository, **kwargs)

class GroundTruthRepositoryTracker(CamDataGeneratorTracker):
    def __init__(self, gt_repo_path, pipeline_repo_path, homography_fn,
                 **kwargs):

        def yield_from_repository():
            yield from ground_truth_legacy.iterate_ground_truth_repository(gt_repo_path, pipeline_repo_path, homography_fn=homography_fn)
            
        super().__init__(yield_from_repository, **kwargs)