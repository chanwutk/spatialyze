#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import json
import os
import pickle
import traceback
import shutil
import socket
import time
import random
import math


from os import environ
from typing import NamedTuple
import datetime

import numpy as np
import torch
import psycopg2

subprocess.Popen('nvidia-smi', shell=True).wait()
process = subprocess.Popen('docker container start spatialyze-gsstore', shell=True)


# In[2]:


hostname = socket.gethostname()
test = hostname.split("-")[-1]
print("test", test)


# In[3]:


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


if is_notebook():
    get_ipython().run_line_magic('cd', '../..')
    from tqdm.notebook import tqdm
    # from ..evaluation.nbutils.report_progress import report_progress
else:
    from tqdm import tqdm
    # from evaluation.ablation.nbutils.report_progress import report_progress


# In[4]:


process.wait()


# In[5]:


from spatialyze.video_processor.camera_config import camera_config
from spatialyze.video_processor.payload import Payload
from spatialyze.video_processor.pipeline import Pipeline
from spatialyze.video_processor.video import Video
from spatialyze.video_processor.metadata_json_encoder import MetadataJSONEncoder


# In[6]:


# Stages
from spatialyze.video_processor.stages.in_view import InView

from spatialyze.video_processor.stages.decode_frame.decode_frame import DecodeFrame as StageDecodeFrame
from spatialyze.video_processor.stages.decode_frame.parallel_decode_frame import ParallelDecodeFrame

from spatialyze.video_processor.stages.detection_2d.detection_2d import Detection2D
from spatialyze.video_processor.stages.detection_2d.yolo_detection import YoloDetection
from spatialyze.video_processor.stages.detection_2d.object_type_filter import ObjectTypeFilter
from spatialyze.video_processor.stages.detection_2d.ground_truth import GroundTruthDetection


# In[7]:


from spatialyze.video_processor.stages.detection_3d.from_detection_2d_and_road import FromDetection2DAndRoad as StageFromDetection2DAndRoad
from spatialyze.video_processor.stages.detection_3d.from_detection_2d_and_depth import FromDetection2DAndDepth as StageFromDetection2DAndDepth

from spatialyze.video_processor.stages.depth_estimation import DepthEstimation

from spatialyze.video_processor.stages.detection_estimation import DetectionEstimation
from spatialyze.video_processor.stages.detection_estimation.segment_mapping import RoadPolygonInfo


# In[8]:


from spatialyze.video_processor.stages.tracking.strongsort import StrongSORT as StageStrongSORT
from spatialyze.video_processor.stages.tracking_2d.strongsort import StrongSORT as StrongSORT2D


# In[9]:


from spatialyze.video_processor.stages.tracking_3d.from_tracking_2d_and_road import FromTracking2DAndRoad
from spatialyze.video_processor.stages.tracking_3d.from_tracking_2d_and_depth import FromTracking2DAndDepth
from spatialyze.video_processor.stages.tracking_3d.tracking_3d import Tracking3DResult, Tracking3D
from spatialyze.video_processor.stages.tracking_3d.from_tracking_2d_and_detection_3d import FromTracking2DAndDetection3D as FromT2DAndD3D

# from spatialyze.video_processor.stages.segment_trajectory import SegmentTrajectory
# from spatialyze.video_processor.stages.segment_trajectory.construct_segment_trajectory import SegmentPoint
# from spatialyze.video_processor.stages.segment_trajectory.from_tracking_3d import FromTracking3D


# In[10]:


# Stream Processing
from spatialyze.video_processor.stream.decode_frame import DecodeFrame
from spatialyze.video_processor.stream.prune_frames import PruneFrames
from spatialyze.video_processor.stream.road_visibility_pruner import RoadVisibilityPruner
from spatialyze.video_processor.stream.mono_depth_estimator import MonoDepthEstimator
from spatialyze.video_processor.stream.yolo import Yolo
from spatialyze.video_processor.stream.object_type_pruner import ObjectTypePruner
from spatialyze.video_processor.stream.from_detection_2d_and_road import FromDetection2DAndRoad
from spatialyze.video_processor.stream.from_detection_2d_and_depth import FromDetection2DAndDepth
from spatialyze.video_processor.stream.exit_frame_sampler import ExitFrameSampler
from spatialyze.video_processor.stream.strongsort import StrongSORT


# In[11]:


# from spatialyze.video_processor.cache import disable_cache
# disable_cache()


# In[12]:


# from spatialyze.video_processor.utils.process_pipeline import format_trajectory, insert_trajectory, get_tracks
from spatialyze.video_processor.utils.prepare_trajectory import prepare_trajectory
from spatialyze.video_processor.utils.insert_trajectory import insert_trajectory
from spatialyze.video_processor.actions.tracking2d_overlay import tracking2d_overlay


# In[13]:


from spatialyze.utils.ingest_road import ingest_road
from spatialyze.database import database, Database
# from spatialyze.legacy.world import empty_world
from spatialyze.utils import F
from spatialyze.predicate import camera, objects, lit, FindAllTablesVisitor, normalize, MapTablesTransformer, GenSqlVisitor
from spatialyze.data_types.camera import Camera as ACamera
from spatialyze.data_types.camera_config import CameraConfig as ACameraConfig


# In[14]:


NUSCENES_PROCESSED_DATA = "NUSCENES_PROCESSED_DATA"
print(NUSCENES_PROCESSED_DATA in os.environ)
print(os.environ['NUSCENES_PROCESSED_DATA'])


# In[15]:


DATA_DIR = os.environ[NUSCENES_PROCESSED_DATA]
# with open(os.path.join(DATA_DIR, "videos", "frames.pkl"), "rb") as f:
#     videos = pickle.load(f)
# with open(os.path.join(DATA_DIR, 'videos', 'videos.json'), 'r') as f:
#     videos = json.load(f)


# In[16]:


with open('./data/evaluation/video-samples/boston-seaport.txt', 'r') as f:
    sampled_scenes = f.read().split('\n')
print(sampled_scenes[0], sampled_scenes[-1], len(sampled_scenes))


# In[17]:


BENCHMARK_DIR = "./outputs/run"


def bm_dir(*args: "str"):
    return os.path.join(BENCHMARK_DIR, *args)


# In[18]:


def get_sql(predicate: "PredicateNode"):
    tables, camera = FindAllTablesVisitor()(predicate)
    tables = sorted(tables)
    mapping = {t: i for i, t in enumerate(tables)}
    predicate = normalize(predicate)
    predicate = MapTablesTransformer(mapping)(predicate)

    t_tables = ''
    t_outputs = ''
    for i in range(len(tables)):
        t_tables += '\n' \
            'JOIN Item_General_Trajectory ' \
            f'AS t{i} ' \
            f'ON Cameras.timestamp <@ t{i}.trajCentroids::period'
        t_outputs += f', t{i}.itemId'

    return f"""
        SELECT Cameras.frameNum {t_outputs}
        FROM Cameras{t_tables}
        WHERE
        {GenSqlVisitor()(predicate)}
    """


# In[19]:


slices = {
    "noopt": (0, 1),
    "inview": (1, 2),
    "objectfilter": (2, 3),
    "geo": (3, 4),
    "de": (4, 5),
    "opt": (5, 6),
    # "optde": (6, 7),
    'dev': (0, 2),
    'freddie': (1, 2),
}


# In[20]:


from spatialyze.video_processor.stream.list_images import ListImages
from spatialyze.video_processor.stream.load_images import LoadImages
from spatialyze.video_processor.stream.detect_topdown_cars import DetectTopDownCars
from spatialyze.video_processor.stream.from_topdown_detection_2d import FromTopDownDetection2D
from spatialyze.video_processor.stream.sort import SORT
from spatialyze.video_processor.stream.topdown_road_visibility_pruner import TopDownRoadVisibilityPruner


# start_date = datetime.datetime(2018, 1, 1, 1, 1, 1, 1, tzinfo=datetime.timezone.utc)
start_date = datetime.datetime(
    year=2018,
    month=8,
    day=27,
    hour=15,
    minute=51,
    second=32,
    microsecond=0
)


class TopDownCameraConfig(NamedTuple):
    tl: tuple[float, float]
    tr: tuple[float, float]
    br: tuple[float, float]
    bl: tuple[float, float]
    camera_id: str
    frame_id: int
    filename: str
    camera_translation: tuple[float, float, float]
    camera_rotation: tuple[float, float, float, float]
    camera_intrinsic: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float]
    ]
    ego_translation: tuple[float, float, float]
    ego_rotation: tuple[float, float, float, float]
    timestamp: datetime.datetime
    camera_heading: float
    ego_heading: float


def make_config(v, idx: int, file: str):
    timestamp = start_date + datetime.timedelta(milliseconds=40 * idx)
    translation = [
        *np.array(v).mean(axis=0).tolist(),
        0
    ]
    return TopDownCameraConfig(
        *v,
        'camera1',
        idx,
        file,
        translation,
        (1, 0, 0, 0),
        [[190, 0, 800], [0, 190, 450], [0, 0, 1]],
        translation,
        (1, 0, 0, 0),
        timestamp,
        0,
        0,
    )


def run_benchmark(pipeline, filename, predicates, run=0, ignore_error=False):
    print(filename)
    metadata_strongsort = {}
    metadata_d2d = {}
    failed_videos = []

    all_metadata = {
        'detection': metadata_d2d,
        'sort': metadata_strongsort,
    }
    # print('# of total    videos:', len(videos))

    # names = set(sampled_scenes[:1])
    # names = set(sampled_scenes)
    # filtered_videos = [
    #     n for n in videos
    #     if n[6:10] in names and 'FRONT' in n # and n.endswith('FRONT')
    # ]
    # N = len(filtered_videos)
    # print('# of filtered videos:', N)

    # # s_from, s_to = slices[test]
    # s_from, s_to = (int(test), int(test) + 1)
    # STEP = math.ceil(N / 10)
    # print('test', test)
    # print('from', s_from*STEP)
    # print('to  ', s_to*STEP)
    # filtered_videos = filtered_videos[s_from*STEP:min(s_to*STEP, N)]
    # print('# of sliced   videos:', len(filtered_videos))
    # ingest_road(database, './data/scenic/road-network/boston-seaport')

    for pre in [*all_metadata.keys(), 'qresult', 'performance', 'failedvideos']:
        p = os.path.join(BENCHMARK_DIR, f"{pre}--{filename}_{run}")
        if os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p)

    def save_perf():
        for n, message in failed_videos:
            p = bm_dir(f'failedvideos--{filename}_{run}', f'{n}.txt')
            with open(p, "w") as f:
                f.write(message)

    filtered_videos = [1]
    for i, name in enumerate(filtered_videos):
        try:
            start_input = time.time()
            video_filename = 'video'
            with open('../data/data/align-out.json', 'r') as f:
                video = json.load(f)

            video = [
                None
                if v is None else
                make_config(v, idx, file)
                for idx, (v, file) in enumerate(zip(video, sorted(os.listdir('../data/frames/main'))))
            ]

            frames = Video('../data/frames/main', video)
            time_input = time.time() - start_input

            start_process = time.time()
            inview = TopDownRoadVisibilityPruner()
            image_files = ListImages()
            # image_files = PruneFrames(inview, image_files)
            d2d = DetectTopDownCars(image_files)
            d3d = FromTopDownDetection2D(d2d)
            tracks = SORT(d3d)
            output = tracks.execute(frames)
            time_process = time.time() - start_process

            times_rquery = []
            predicate = predicates[0]
            start_rquery = time.time()
            database.reset(True)

            # Ingest Trackings
            for track in output:
                obj_id = track[0].object_id
                trajectory = prepare_trajectory(
                    name,
                    obj_id,
                    track,
                    frames.camera_configs
                )
                if trajectory:
                    insert_trajectory(database, *trajectory)

            # Ingest Camera
            accs: 'ACameraConfig' = []
            camera_id = ''
            for idx, cc in enumerate(frames.interpolated_frames):
                if cc is None:
                    continue

                camera_id = cc.camera_id
                acc = ACameraConfig(
                    frame_id=cc.frame_id,
                    frame_num=idx,
                    filename=cc.filename,
                    camera_translation=cc.camera_translation,
                    camera_rotation=cc.camera_rotation,
                    camera_intrinsic=cc.camera_intrinsic,
                    ego_translation=cc.ego_translation,
                    ego_rotation=cc.ego_rotation,
                    timestamp=cc.timestamp,
                    cameraHeading=cc.camera_heading,
                    egoHeading=cc.ego_heading,
                )
                accs.append(acc)
            camera = ACamera(accs, camera_id)
            database.insert_camera(camera)

            query = get_sql(predicate)
            qresult = database.execute(query)

            p = bm_dir(f"qresult--{filename}_{run}", f"{name}-{i}.json")
            with open(p, 'w') as f:
                json.dump(qresult, f, indent=1)
            time_rquery = time.time() - start_rquery
            times_rquery.append(time_rquery)
            # runtime_query.append({'name': name, 'predicate': i, 'runtime': time_rquery})

            # save video
            start_video = time.time()
            tracking2d_overlay(output, frames, '.')
            time_video = time.time() - start_video
            # runtime_video.append({'name': name, 'runtime': time_video})

            perf = []
    #         for stage in pipeline.stages:
    #             benchmarks = [*filter(
    #                 lambda x: video['filename'] in x['name'],
    #                 stage.benchmark
    #             )]
    #             assert len(benchmarks) == 1
    #             perf.append({
    #                 'stage': stage.classname(),
    #                 'benchmark': benchmarks[0]
    #             })

    #             for bm in getattr(stage, '_benchmark', []):
    #                 if video['filename'] in bm['name']:
    #                     perf.append({
    #                         'stage': stage.classname(),
    #                         'addition': True,
    #                         'benchmark': bm,
    #                     })

            perf.append({
                'stage': 'ingest',
                'benchmark': {
                    'name': name,
                    'runtime': time_input
                }
            })
            perf.append({
                'stage': 'process',
                'benchmark': {
                    'name': name,
                    'runtime': time_process
                }
            })
            perf.append({
                'stage': 'save',
                'benchmark': {
                    'name': name,
                    'runtime': time_video
                }
            })
            for i, time_rquery in enumerate(times_rquery):
                perf.append({
                    'stage': 'query',
                    'benchmark': {
                        'name': name,
                        'predicate': i,
                        'runtime': time_rquery
                    }
                })
            p = bm_dir(f'performance--{filename}_{run}', f'{name}.json')
            with open(p, "w") as f:
                json.dump(perf, f, indent=1)
        except Exception as e:
            if ignore_error:
                message = str(traceback.format_exc())
                failed_videos.append((name, message))
                print(video_filename)
                print(e)
                print(message)
                print("------------------------------------------------------")
                print()
                print()
            else:
                raise e

        if len(metadata_d2d) % 10 == 0:
            save_perf()
    save_perf()


# In[21]:


def create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=False,
    geo_depth=True,
    detection_estimation=True,
    strongsort=False,
    ss_update_when_skip=True,
    ss_cache=True,
):
    pipeline = Pipeline()

    # In-View Filter
    if in_view:
        # TODO: view angle and road type should depends on the predicate
        pipeline.add_filter(InView(50, predicate=predicate))

    # Decode
    pipeline.add_filter(ParallelDecodeFrame())

    # 2D Detection
    if groundtruth_detection:
        with open(os.path.join(DATA_DIR, 'annotation_partitioned.pkl'), 'rb') as f:
            df_annotations = pickle.load(f)
        pipeline.add_filter(GroundTruthDetection(df_annotations))
    else:
        pipeline.add_filter(YoloDetection())

    # Object Filter
    if object_filter:
        pipeline.add_filter(ObjectTypeFilter(predicate=predicate))

    # 3D Detection
    if geo_depth:
        pipeline.add_filter(StageFromDetection2DAndRoad())
    else:
        pipeline.add_filter(DepthEstimation())
        pipeline.add_filter(StageFromDetection2DAndDepth())

    # Detection Estimation
    if detection_estimation:
        pipeline.add_filter(DetectionEstimation())

    # Tracking
    pipeline.add_filter(StrongSORT2D(
        # method='update-empty' if ss_update_when_skip else 'increment-ages',
        method='update-empty',
        cache=ss_cache,
    ))

    pipeline.add_filter(FromT2DAndD3D())
    # if geo_depth:
    #     pipeline.add_filter(FromTracking2DAndRoad())
    # else:
    #     pipeline.add_filter(FromTracking2DAndDepth())

    # Segment Trajectory
    # pipeline.add_filter(FromTracking3D())

    return pipeline


# In[22]:


p_noSSOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False,
    ss_cache=False,
)

p_noOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False
)

p_inview = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=False,
    geo_depth=False,
    detection_estimation=False
)

p_objectFilter = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=True,
    geo_depth=False,
    detection_estimation=False
)

p_geo = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=True,
    detection_estimation=False
)

p_de = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=True
)

p_deIncr = lambda predicate: create_pipeline(
    predicate,
    in_view=False,
    object_filter=False,
    geo_depth=False,
    detection_estimation=True,
    ss_update_when_skip=False,
)

p_opt = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=False
)

p_optDe = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=True
)

p_optIncr = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=False,
    ss_update_when_skip=False,
)

p_optDeIncr = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    geo_depth=True,
    detection_estimation=True,
    ss_update_when_skip=False,
)

p_gtOpt = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=True,
    geo_depth=True,
    detection_estimation=False
)

p_gtOptDe = lambda predicate: create_pipeline(
    predicate,
    in_view=True,
    object_filter=True,
    groundtruth_detection=True,
    geo_depth=True,
    detection_estimation=True
)

pipelines = {
    "nossopt": p_noSSOpt,
    "noopt": p_noOpt,
    "inview": p_inview,
    "objectfilter": p_objectFilter,
    "geo": p_geo,
    "de": p_de,
    # "deincr": p_deIncr,
    "opt": p_opt,
    # "optincr": p_optIncr,
    "optde": p_optDe,
    # "optdeincr": p_optDeIncr,

    # "gtopt": p_gtOpt,
    # "gtoptde": p_gtOptDe
}


# In[23]:


# if test == 'dev':
#     test = 'opt'


# In[24]:


def run(__test):
    obj1 = objects[0]
    cam = camera
    pred2 = (
        ((obj1.type == 'car') | (obj1.type == 'truck')) &
        F.contained(obj1.trans@cam.time, 'intersection')
    )

    p2 = pipelines[__test](pred2)

    print('Pipeline P2:')
    for s in p2.stages:
        print(' -', s)
    run_benchmark(p2, 'q2-' + __test, [pred2], run=1, ignore_error=True)


# In[25]:


# tests = ['optde', 'de', 'noopt', 'inview', 'objectfilter', 'geo', 'opt']
# tests = ['de', 'noopt', 'inview', 'objectfilter']
tests = ['noopt']
# random.shuffle(tests)

for _test in tests:
    assert isinstance(pipelines[_test](lit(True)), Pipeline)

for idx, _test in enumerate(tests):
    print(f'----------- {idx} / {len(tests)} --- {_test} -----------')
    done = False
    retry = 0
    while not done and retry < 5:
        # try:
        run(_test)
        done = True
        # except Exception as e:
        #     print(type(e))
        #     print(e)
        #     print('retrying...')
        #     time.sleep(60)
        #     retry += 1
        #     with open(os.path.join(BENCHMARK_DIR, f'exception--bm{test}-t{_test}-r{retry}'), 'w') as f:
        #         f.write(str(e))


# In[ ]:


# run(test)


# In[ ]:


# if test == 'opt':
#     run('optde')


# In[ ]:


if not is_notebook():
    subprocess.Popen('sudo shutdown -h now', shell=True)

