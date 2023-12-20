import json
import pickle
import os
import numpy as np

from spatialyze.video_processor.cache import disable_cache
from spatialyze.video_processor.metadata_json_encoder import MetadataJSONEncoder
from spatialyze.video_processor.stages.tracking_3d.tracking_3d import Tracking3DResult
from spatialyze.video_processor.stream.strongsort import TrackingResult
from spatialyze.world import _execute

from common import build_filter_world


OUTPUT_DIR = './data/pipeline/test-results'
disable_cache()


def trackid(track: list[TrackingResult]):
    return track[0].object_id


def frameidx(track: TrackingResult):
    return track.detection_id.frame_idx


def test_optimized_workflow():
    world = build_filter_world(pkl=True)
    objects, trackings = _execute(world)

    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-trackings.json'), 'w') as f:
    #     json.dump(trackings, f, indent=1, cls=MetadataJSONEncoder)
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-trackings.pkl'), 'wb') as f:
    #     pickle.dump(trackings, f)
    
    with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-trackings.pkl'), 'rb') as f:
        trackings_groundtruth = pickle.load(f)
    
    for filename, tgs in trackings_groundtruth.items():
        assert filename in trackings, (filename, trackings.keys())
        tps = trackings[filename]
        assert len(tps) == len(tgs), (len(tps), len(tgs))
        for idx, (tp, tg) in enumerate(zip(sorted(tps, key=trackid), sorted(tgs, key=trackid))):
            assert len(tp) == len(tg), (idx, len(tp), len(tg))
            for p, g in zip(sorted(tp, key=frameidx), sorted(tg, key=frameidx)):
                assert isinstance(p, TrackingResult), (p, type(p))
                assert isinstance(g, TrackingResult), (g, type(g))
                assert tuple(p.detection_id) == tuple(g.detection_id), (p.detection_id, g.detection_id)
                assert p.object_id == g.object_id, (p.object_id, g.object_id)
                assert p.object_type == g.object_type, (p.object_type, g.object_type)
                assert p.timestamp == g.timestamp, (p.timestamp, g.timestamp)
                assert np.allclose([p.confidence], [g.confidence], atol=0.001, rtol=0), (p.confidence, g.confidence)
                assert np.allclose(p.bbox.detach().cpu().numpy(), g.bbox.detach().cpu().numpy(), atol=0.001, rtol=0), (p.bbox, g.bbox)
    
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects.json'), 'w') as f:
    #     json.dump(objects, f, indent=1)
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects.pkl'), 'wb') as f:
    #     pickle.dump(objects, f)
    
    with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects.pkl'), 'rb') as f:
        objects_groundtruth = pickle.load(f)
    
    assert len(objects) == len(objects_groundtruth), (len(objects), len(objects_groundtruth))
    for filename, ogs in objects_groundtruth.items():
        assert filename in objects, (filename, objects.keys())
        ops = objects[filename]
        assert len(ops) == len(ogs), (len(ops), len(ogs))
        for p, og in zip(sorted(ops), sorted(ogs)):
            assert tuple(p) == tuple(og), (p, og)

    world._objects, world._trackings = objects, trackings
    objects2 = world.getObjects()

    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects2.json'), 'w') as f:
    #     json.dump(objects2, f, indent=1)
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects2.pkl'), 'wb') as f:
    #     pickle.dump(objects2, f)
    
    with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects2.pkl'), 'rb') as f:
        objects2_groundtruth = pickle.load(f)
    
    for o2, og in zip(sorted(objects2), sorted(objects2_groundtruth)):
        assert o2.id == og.id, (o2.id, og.id)
        assert o2.type == og.type, (o2.type, og.type)
        assert np.allclose(o2.track, og.track, atol=0.001, rtol=0), (o2.track, og.track)
        assert np.allclose(o2.bboxes, og.bboxes, atol=2, rtol=0), (o2.bboxes, og.bboxes)
        assert np.allclose(o2.frame_ids, og.frame_ids), (o2.frame_ids, og.frame_ids)
        assert o2.camera_id == og.camera_id, (o2.camera_id, og.camera_id)
    
    world.saveVideos('./outputs', addBoundingBoxes=True)
    assert os.path.exists('./outputs/scene-0655-CAM_FRONT-result.mp4')
    assert os.path.exists('./outputs/scene-0757-CAM_FRONT-result.mp4')


def test_optimized_workflow_alt():
    world = build_filter_world(pkl=True, alt_tracker=True)
    objects, trackings = _execute(world)

    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-trackings-alt.json'), 'w') as f:
    #     json.dump(trackings, f, indent=1, cls=MetadataJSONEncoder)
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-trackings-alt.pkl'), 'wb') as f:
    #     pickle.dump(trackings, f)
    
    with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-trackings-alt.pkl'), 'rb') as f:
        trackings_groundtruth = pickle.load(f)
    
    for filename, tgs in trackings_groundtruth.items():
        assert filename in trackings, (filename, trackings.keys())
        tps = trackings[filename]
        assert len(tps) == len(tgs), (len(tps), len(tgs))
        for idx, (tp, tg) in enumerate(zip(sorted(tps, key=trackid), sorted(tgs, key=trackid))):
            assert len(tp) == len(tg), (idx, len(tp), len(tg))
            for p, g in zip(sorted(tp, key=frameidx), sorted(tg, key=frameidx)):
                assert isinstance(p, TrackingResult), (p, type(p))
                assert isinstance(g, TrackingResult), (g, type(g))
                assert tuple(p.detection_id) == tuple(g.detection_id), (p.detection_id, g.detection_id)
                assert p.object_id == g.object_id, (p.object_id, g.object_id)
                assert p.object_type == g.object_type, (p.object_type, g.object_type)
                assert p.timestamp == g.timestamp, (p.timestamp, g.timestamp)
                assert np.allclose([p.confidence], [g.confidence], atol=0.001, rtol=0), (p.confidence, g.confidence)
                assert np.allclose(p.bbox.detach().cpu().numpy(), g.bbox.detach().cpu().numpy(), atol=0.001, rtol=0), (p.bbox, g.bbox)
    
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects-alt.json'), 'w') as f:
    #     json.dump(objects, f, indent=1)
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects-alt.pkl'), 'wb') as f:
    #     pickle.dump(objects, f)
    
    with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects-alt.pkl'), 'rb') as f:
        objects_groundtruth = pickle.load(f)
    
    assert len(objects) == len(objects_groundtruth), (len(objects), len(objects_groundtruth))
    for filename, ogs in objects_groundtruth.items():
        assert filename in objects, (filename, objects.keys())
        ops = objects[filename]
        assert len(ops) == len(ogs), (len(ops), len(ogs))
        for p, og in zip(sorted(ops), sorted(ogs)):
            assert tuple(p) == tuple(og), (p, og)

    world._objects, world._trackings = objects, trackings
    objects2 = world.getObjects()

    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects2-alt.json'), 'w') as f:
    #     json.dump(objects2, f, indent=1)
    # with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects2-alt.pkl'), 'wb') as f:
    #     pickle.dump(objects2, f)
    
    with open(os.path.join(OUTPUT_DIR, 'optimized-workflow-objects2-alt.pkl'), 'rb') as f:
        objects2_groundtruth = pickle.load(f)
    
    for o2, og in zip(sorted(objects2), sorted(objects2_groundtruth)):
        assert o2.id == og.id, (o2.id, og.id)
        assert o2.type == og.type, (o2.type, og.type)
        assert np.allclose(o2.track, og.track, atol=0.001, rtol=0), (o2.track, og.track)
        assert np.allclose(o2.bboxes, og.bboxes, atol=2, rtol=0), (o2.bboxes, og.bboxes)
        assert np.allclose(o2.frame_ids, og.frame_ids), (o2.frame_ids, og.frame_ids)
        assert o2.camera_id == og.camera_id, (o2.camera_id, og.camera_id)
    
    world.saveVideos('./outputs', addBoundingBoxes=True)
    assert os.path.exists('./outputs/scene-0655-CAM_FRONT-result.mp4')
    assert os.path.exists('./outputs/scene-0757-CAM_FRONT-result.mp4')