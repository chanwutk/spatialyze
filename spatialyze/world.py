import datetime
from typing import Type

import torch

from .data_types.query_result import QueryResult
from .database import METADATA_TABLE, Database
from .database import database as default_database
from .geospatial_video import GeospatialVideo
from .predicate import (
    BoolOpNode,
    CameraTableNode,
    ObjectTableNode,
    PredicateNode,
    is_detection_only,
    lit,
)
from .road_network import RoadNetwork
from .utils.F.road_segment import road_segment
from .utils.get_object_list import get_object_list
from .utils.ingest_road import create_tables, drop_tables
from .utils.save_video_util import save_video_util
from .video_processor.stream.data_types import Detection2D, Detection3D, Skip
from .video_processor.stream.decode_frame import DecodeFrame
from .video_processor.stream.from_detection_2d_and_depth import FromDetection2DAndDepth
from .video_processor.stream.from_detection_2d_and_road import FromDetection2DAndRoad
from .video_processor.stream.mono_depth_estimator import MonoDepthEstimator
from .video_processor.stream.object_type_pruner import ObjectTypePruner
from .video_processor.stream.prefilter import Prefilter
from .video_processor.stream.prune_frames import PruneFrames
from .video_processor.stream.road_visibility_pruner import RoadVisibilityPruner

# stream
from .video_processor.stream.stream import Stream
from .video_processor.stream.strongsort import StrongSORT, TrackingResult
from .video_processor.stream.yolo import Yolo
from .video_processor.types import DetectionId
from .video_processor.utils.insert_detections import insert_detections
from .video_processor.utils.insert_trajectory import insert_trajectory
from .video_processor.utils.prepare_trajectory import prepare_trajectory
from .video_processor.video import Video

TrackingResults = list[TrackingResult]


class World:
    def __init__(
        self,
        database: "Database | None" = None,
        predicates: "list[PredicateNode] | None" = None,
        videos: "list[GeospatialVideo] | None" = None,
        geogConstructs: "list[RoadNetwork] | None" = None,
        detector: Type[Stream[Detection2D]] | None = None,
        tracker: Type[Stream[TrackingResults]] | None = None,
        processor: Stream[TrackingResults] | None = None,
    ):
        self._database = database or default_database
        self._predicates = predicates or []
        self._videos = videos or []
        self._geogConstructs = geogConstructs or []
        self._objectCounts = 0
        self._objects: "dict[str, list[QueryResult]] | None" = None
        self._trackings: "dict[str, list[TrackingResults]] | None" = None
        self._detector: tuple[Type[Stream[Detection2D]]] = (detector or Yolo,)
        self._tracker: tuple[Type[Stream[TrackingResults]]] = (tracker or StrongSORT,)
        self._processor: Stream[TrackingResults] | None = processor
        # self._cameraCounts = 0

    @property
    def predicates(self) -> "PredicateNode":
        if len(self._predicates) == 0:
            return lit(True)
        if len(self._predicates) == 1:
            return self._predicates[0]
        return BoolOpNode("and", self._predicates)

    def filter(self, predicate: "PredicateNode") -> "World":
        self._predicates.append(predicate)
        self._objects, self._trackings = None, None
        return self

    def addVideo(self, video: "GeospatialVideo") -> "World":
        self._videos.append(video)
        self._objects, self._trackings = None, None
        return self

    def addGeogConstructs(self, geogConstructs: "RoadNetwork") -> "World":
        self._geogConstructs.append(geogConstructs)
        self._objects, self._trackings = None, None
        return self

    def object(self, index: "int | None" = None):
        if index is not None:
            return ObjectTableNode(index)

        node = ObjectTableNode(self._objectCounts)
        self._objectCounts += 1
        return node

    def camera(self) -> "CameraTableNode":
        return CameraTableNode(0)

    def geogConstruct(self, type: "str"):
        return road_segment(type)

    def saveVideos(self, outputDir: "str", addBoundingBoxes: "bool" = False):
        if self._objects is None or self._trackings is None:
            self._objects, self._trackings = _execute(self)
        return save_video_util(
            self._objects,
            self._trackings,
            outputDir,
            addBoundingBoxes,
        )

    def getObjects(self):
        """
        Returns a list of moveble objects, with each object tuple containing:
        - object id
        - object type
        - trajectory
        - bounding boxes
        - frame IDs
        - camera id
        """
        if self._objects is None or self._trackings is None:
            self._objects, self._trackings = _execute(self)

        return get_object_list(self._objects, self._trackings)


BATCH_SIZE = 2048


def _execute(world: "World", optimization=True):
    database = world._database
    (detector,) = world._detector
    (tracker,) = world._tracker
    processor = world._processor

    # add geographic constructs
    drop_tables(database)
    create_tables(database)
    for gc in world._geogConstructs:
        gc.ingest(database)

    temporal = not is_detection_only(world.predicates)

    qresults: dict[str, list[QueryResult]] = {}
    vresults: dict[str, list[TrackingResults]] = {}
    for v in world._videos:
        # reset database
        database.reset()
        database.insert_camera(v.camera)

        decode = DecodeFrame()
        if v.keep is not None:
            prefilter = Prefilter(v.keep)
            decode = PruneFrames(prefilter, decode)
        if optimization:
            inview = RoadVisibilityPruner(distance=50, predicate=world.predicates)
            decode = PruneFrames(inview, decode)
        d2ds = detector(decode)

        if optimization:
            d2ds = ObjectTypePruner(d2ds, predicate=world.predicates)
            d3ds = FromDetection2DAndRoad(d2ds)
            # if temporal and all(t in ["car", "truck"] for t in d2ds.types):
            #     efs = ExitFrameSampler(d3ds)
            #     d3ds = PruneFrames(efs, d3ds)
        else:
            depths = MonoDepthEstimator(decode)
            d3ds = FromDetection2DAndDepth(d2ds, depths)
        t3ds = processor or tracker(d3ds, decode)

        # execute pipeline
        video = Video(v.video, v.camera)
        database.update(f"INSERT INTO {METADATA_TABLE} (fps) VALUES ({video.fps})")
        process = _track(t3ds) if temporal else _detect(d3ds)
        vresults[v.video] = process(video, database)

        assert all(idx == cc.frame_num for idx, cc in enumerate(v.camera)), [
            cc.frame_num for cc in v.camera
        ]
        qresults[v.video] = database.predicate(world.predicates, temporal)
    return qresults, vresults


def _track(processor: Stream[TrackingResults]):
    def _(video: Video, database: Database):
        vresults: list[TrackingResults] = []
        for track in processor.iterate(video):
            assert not isinstance(track, Skip)
            vresults.append(track)

            obj_id = track[0].object_id
            trajectory = prepare_trajectory(obj_id, track, video.camera_configs)
            if trajectory:
                insert_trajectory(database, trajectory)
        assert processor.ended()
        return vresults

    return _


def _detect(processor: Stream[Detection3D]):
    def _(video: Video, database: Database):
        camera_id = video[0].camera_id
        clss: list[str] | None = None
        vresults: list[TrackingResults] = []
        for idx, detections in enumerate(processor.iterate(video)):
            if isinstance(detections, Skip) or len(detections[0]) == 0:
                continue

            timestamp = video[idx].timestamp
            insert_detections(database, detections, camera_id, idx, timestamp)

            if clss is None:
                clss = detections[1]
                assert isinstance(clss, list)

            vresults.extend(
                tracking_result(det, did, clss, timestamp) for det, _, did in zip(*detections)
            )

        assert processor.ended()
        return vresults

    return _


def tracking_result(
    det: torch.Tensor,
    did: DetectionId,
    clss: list[str],
    timestamp: datetime.datetime,
) -> TrackingResults:
    oid = f"{did.frame_idx}__{did.obj_order}"
    conf = float(det[4])
    cls = clss[int(det[5])]
    tr = TrackingResult(did, oid, conf, det, cls, timestamp)
    return [tr]
