import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from ..camera_config import CameraConfig
from ..modules.yolo_tracker.trackers.multi_tracker_zoo import StrongSORT as _StrongSORT
from ..modules.yolo_tracker.trackers.multi_tracker_zoo import create_tracker
from ..modules.yolo_tracker.trackers.strong_sort.sort.track import Track
from ..modules.yolo_tracker.yolov5.utils.torch_utils import select_device
from ..types import DetectionId
from ..video import Video
from .data_types import Detection2D, Detection3D, Skip
from .stream import Stream

FILE = Path(__file__).resolve()
SPATIALYZE = FILE.parent.parent.parent.parent
WEIGHTS = SPATIALYZE / "weights"
REID_WEIGHTS = WEIGHTS / "osnet_x0_25_msmt17.pt"
EMPTY_DETECTION = torch.Tensor(0, 6)


@dataclass
class TrackingResult:
    detection_id: DetectionId
    object_id: int
    confidence: float | np.float32
    bbox: torch.Tensor
    object_type: str
    timestamp: datetime.datetime
    next: "TrackingResult | None" = field(default=None, compare=False, repr=False)
    prev: "TrackingResult | None" = field(default=None, compare=False, repr=False)


class StrongSORT(Stream[list[TrackingResult]]):
    def __init__(
        self, detections: Stream[Detection2D] | Stream[Detection3D], frames: Stream[npt.NDArray]
    ):
        self.detection2ds = detections
        self.frames = frames

    def _stream(self, video: Video):
        device = select_device()
        strongsort = create_tracker("strongsort", REID_WEIGHTS, device, False)
        assert isinstance(strongsort, _StrongSORT)
        assert hasattr(strongsort, "tracker")
        assert hasattr(strongsort.tracker, "camera_update")
        assert hasattr(strongsort, "model")
        assert hasattr(strongsort.model, "warmup")
        curr_frame, prev_frame = None, None
        deleted_tracks_idx = 0
        with torch.no_grad():
            strongsort.model.warmup()
            # init_end = time.time()

            # update_time = 0
            # skip_time = 0
            # tracking_start = time.time()
            # assert len(detections) == len(images)
            saved_detections: list[dict[int, torch.Tensor]] = []
            clss: list[str] | None = None
            for detection, im0s in zip(self.detection2ds.stream(video), self.frames.stream(video)):
                assert not isinstance(im0s, Skip), type(im0s)
                im0 = im0s.copy()
                curr_frame = im0

                # update_start = time.time()
                # Always do camera update
                if prev_frame is not None and curr_frame is not None:
                    strongsort.tracker.camera_update(prev_frame, curr_frame, cache=True)
                prev_frame = curr_frame
                # update_time += time.time() - update_start

                det, dids = EMPTY_DETECTION, []
                if not isinstance(detection, Skip) and len(detection[0]) > 0:
                    det, _classes, dids = detection
                    if clss is None:
                        clss = _classes
                det = det.cpu()
                strongsort.update(det, dids, im0)
                saved_detections.append({int(did.obj_order): dt for dt, did in zip(det, dids)})

                deleted_tracks = strongsort.tracker.deleted_tracks
                while deleted_tracks_idx < len(deleted_tracks):
                    yield _process_track(
                        deleted_tracks[deleted_tracks_idx],
                        saved_detections,
                        clss,
                        video.camera_configs,
                    )
                    deleted_tracks_idx += 1
                # skip_time += time.time() - skip_start
            for track in strongsort.tracker.tracks:
                yield _process_track(track, saved_detections, clss, video.camera_configs)
            # tracking_end = time.time()

        # self.ss_benchmark.append({
        #     'file': payload.video.videofile,
        #     'load_data': load_data_end - load_data_start,
        #     'init': init_end - init_start,
        #     'tracking': tracking_end - tracking_start,
        #     'skip': skip_time,
        #     'update_camera': update_time,
        #     'postprocess': postprocess_end - postprocess_start,
        # })


def _process_track(
    track: Track,
    detections: list[dict[int, torch.Tensor]],
    clss: list[str] | None,
    camera_configs: list[CameraConfig],
):
    tid = track.track_id
    assert isinstance(tid, int), type(tid)

    clss = clss or []

    def tracking_result(did_conf: tuple[DetectionId, float]):
        did, conf = did_conf
        fid, oid = did
        assert isinstance(oid, int), type(oid)
        bbox = detections[fid][oid]
        cls = int(bbox[5])
        return TrackingResult(
            did, tid, conf, detections[fid][oid], clss[cls], camera_configs[fid].timestamp
        )

    # Sort track by frame idx
    _track = map(tracking_result, zip(track.detection_ids, track.confs))
    _track = sorted(_track, key=lambda d: d.detection_id.frame_idx)

    # Link track
    for before, after in zip(_track[:-1], _track[1:]):
        before.next = after
        after.prev = before

    return _track