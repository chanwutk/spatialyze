from typing import Literal

import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from postgis import MultiPoint
from psycopg2 import sql
from pyquaternion import Quaternion

from ....database import database
from ....predicate import (
    ArrayNode,
    BaseTransformer,
    BinOpNode,
    BoolOpNode,
    CallNode,
    CameraTableNode,
    CastNode,
    CompOpNode,
    LiteralNode,
    ObjectTableNode,
    PredicateNode,
    TableAttrNode,
    TableNode,
    UnaryOpNode,
    Visitor,
    lit,
)
from ....utils import F
from ...payload import Payload
from ...video import Video
from ..stage import Stage

OTHER_ROAD_TYPES = {
    "roadsection": ["road", "intersection"],
    "lane": ["lanegroup", "road", "roadsection", "intersection"],
    "lanesection": ["lanegroup", "road", "roadsection", "intersection"],
    "lanegroup": ["road", "roadsection", "intersection"],
    "intersection": ["lanegroup", "lane", "lanesection"],
}


class InView(Stage):
    def __init__(
        self,
        distance: float,
        roadtypes: "str | list[str] | None" = None,
        predicate: "PredicateNode | None" = None,
    ):
        super().__init__()
        self.distance = distance
        assert (
            roadtypes is not None or predicate is not None
        ), "At least one of roadtypes or predicate must be specified"
        if roadtypes is not None:
            assert predicate is None, "Can only except either segment_type or predicate"
            self.roadtypes = roadtypes if isinstance(roadtypes, list) else [roadtypes]
            self.predicate = None
        elif predicate is not None:
            assert roadtypes is None, "Can only except either segment_type or predicate"
            self.roadtypes, self.predicate_str = create_inview_predicate(predicate)
            self.predicate = eval(str(self.predicate_str))

    def __repr__(self) -> str:
        return f"InView(distance={self.distance}, roadtype={self.roadtypes is not None and self.roadtypes}, predicate={hasattr(self, 'predicate_str') and self.predicate_str})"

    def _run(self, payload: "Payload") -> "tuple[bitarray, None]":
        indices, view_areas = get_views(payload.video, self.distance)

        keep = bitarray(len(payload.keep))
        keep.setall(1)
        if self.predicate is None:
            results = database.execute(
                sql.SQL(
                    """
            SELECT index
            FROM UNNEST (
                {view_areas},
                {indices}::int[]
            ) AS ViewArea(points, index)
            JOIN SegmentPolygon ON ST_Intersects(ST_ConvexHull(points), elementPolygon)
            WHERE {segment_type}
            """
                ).format(
                    view_areas=sql.Literal(view_areas),
                    indices=sql.Literal(indices),
                    segment_type=sql.SQL(" OR ".join(map(roadtype, self.roadtypes))),
                )
            )

            keep.setall(0)
            for (index,) in results:
                keep[index] = 1
        elif self.predicate is False:
            keep.setall(0)
        elif self.predicate is not True:
            exists = sql.SQL(
                """
            EXISTS (
                SELECT *
                FROM SegmentPolygon
                WHERE ST_Intersects(ST_ConvexHull(points), elementPolygon)
                AND {rt}
            )
            """
            )
            results = database.execute(
                sql.SQL(
                    """
            SELECT index, {exists}
            FROM UNNEST (
                {view_areas},
                {indices}::int[]
            ) AS ViewArea(points, index)
            """
                ).format(
                    view_areas=sql.Literal(view_areas),
                    indices=sql.Literal(indices),
                    exists=sql.SQL(",").join(
                        exists.format(rt=sql.Identifier(roadtype(st))) for st in self.roadtypes
                    ),
                )
            )

            keep.setall(0)
            for index, *encoded_segment_types in results:
                segment_types = {
                    st for st, encoded in zip(self.roadtypes, encoded_segment_types) if encoded
                }
                if self.predicate(segment_types):
                    keep[index] = 1

        return keep, None


def get_views(video: "Video", distance: "float" = 100):
    w, h = video.dimension
    Z = distance
    view_vertices_2d = np.array(
        [
            # 4 corners of the image frame
            (w, h, 1),
            (w, 0, 1),
            (0, h, 1),
            (0, 0, 1),
            # camera position
            (0, 0, 0),
        ]
    ).T
    assert view_vertices_2d.shape == (3, 5), view_vertices_2d.shape

    [[fx, s, x0], [_, fy, y0], [_, _, _]] = video.interpolated_frames[0].camera_intrinsic

    # 3x3 matrix to convert points from pixel-coordinate to camera-coordinate
    pixel2camera = Z * np.array(
        [
            [1 / fx, -s / (fx * fy), (s * y0 / (fx * fy)) - (x0 / fx)],
            [0, 1 / fy, -y0 / fy],
            [0, 0, 1],
        ]
    )
    assert pixel2camera.shape == (3, 3), pixel2camera.shape

    view_vertices_from_camera = pixel2camera @ view_vertices_2d
    assert view_vertices_from_camera.shape == (3, 5), view_vertices_from_camera.shape

    extrinsics: "list[npt.NDArray]" = []
    indices: "list[int]" = []
    # for i, (k, f) in enumerate(zip(keep, video.interpolated_frames)):
    #     if not k and skip:
    #         continue
    for i, f in enumerate(video.interpolated_frames):
        rotation = Quaternion(f.camera_rotation)
        rotation_matrix = rotation.unit.rotation_matrix
        assert rotation_matrix.shape == (3, 3), rotation_matrix.shape

        # 3x4 matrix to convert points from camera-coordinate to world-coordinate
        translation = np.array(f.camera_translation)[np.newaxis].T
        extrinsic = np.hstack((rotation_matrix, translation))
        assert extrinsic.shape == (3, 4), extrinsic.shape

        extrinsics.append(extrinsic)
        indices.append(i)

    N = len(extrinsics)

    # add 1 to the last row
    view_vertices_from_camera = np.concatenate(
        (
            view_vertices_from_camera,
            np.ones_like(view_vertices_from_camera[:1]),
        )
    )

    _extrinsics = np.stack(extrinsics, dtype=np.floating)
    assert _extrinsics.shape == (N, 3, 4), _extrinsics.shape

    # convert 4 corner points from pixel-coordinate to world-coordinate
    view_area_3ds = _extrinsics @ view_vertices_from_camera
    assert view_area_3ds.shape == (N, 3, 5), view_area_3ds.shape

    # project view_area to 2D from top-down view
    view_area_2ds = view_area_3ds[:, :2].swapaxes(1, 2)
    assert view_area_2ds.shape == (N, 5, 2), view_area_2ds.shape

    assert any(
        np.array_equal(view_area_3ds[n, :2, i], view_area_2ds[n, i])
        for n in range(N)
        for i in range(5)
    ), (view_area_3ds, view_area_2ds)

    view_areas: "list[MultiPoint]" = []
    for i, view_area_2d in zip(indices, view_area_2ds):
        view_area = MultiPoint(view_area_2d.tolist())
        view_areas.append(view_area)

    return indices, view_areas


def roadtype(t: "str"):
    return f"__roadtype__{t}__"


ANNIHILATORS: "dict[Literal['and', 'or'], bool]" = {
    "and": False,
    "or": True,
}

IS_ROADTYPE = F.is_roadtype().fn
IS_OTHER_ROADTYPE = F.is_other_roadtype().fn
IGNORE_ROADTYPE = F.ignore_roadtype().fn


class KeepOnlyRoadTypePredicates(BaseTransformer):
    def visit_ArrayNode(self, node: ArrayNode):
        return F.ignore_roadtype()

    def visit_CompOpNode(self, node: CompOpNode):
        return F.ignore_roadtype()

    def visit_BinOpNode(self, node: BinOpNode):
        return F.ignore_roadtype()

    def visit_BoolOpNode(self, node: BoolOpNode):
        visited = super().visit_BoolOpNode(node)
        assert isinstance(visited, BoolOpNode), visited
        annihilator = ANNIHILATORS[node.op]
        # print(visited.op, annihilator)
        # print(visited)
        if any(isinstance(e, LiteralNode) and e.value == annihilator for e in visited.exprs):
            # print('annihilated', visited)
            return lit(annihilator)
        if all(isinstance(e, LiteralNode) for e in visited.exprs):
            exprs = visited.exprs
            assert all(isinstance(e, LiteralNode) for e in exprs), exprs
            assert len({e.value for e in exprs if isinstance(e, LiteralNode)}) == 1, visited.exprs
            # print('all', visited)
            e = exprs[0]
            assert isinstance(e, LiteralNode), e
            return lit(e.value)
        exprs = [e for e in visited.exprs if not isinstance(e, LiteralNode)]
        return BoolOpNode(visited.op, exprs) if len(exprs) > 1 else exprs[0]

    def visit_UnaryOpNode(self, node: UnaryOpNode):
        visited = super().visit_UnaryOpNode(node)
        assert isinstance(visited, UnaryOpNode), visited
        # print(node.op, visited)

        if node.op == "neg":
            return visited.expr

        if isinstance(visited.expr, LiteralNode):
            assert isinstance(visited.expr.value, bool), visited.expr
            # print('inverted', visited)
            return lit(not visited.expr.value)

        return visited

    def visit_LiteralNode(self, node: LiteralNode):
        if isinstance(node.value, bool):
            return node
        return F.ignore_roadtype()

    def visit_TableAttrNode(self, node: TableAttrNode):
        return F.ignore_roadtype()

    def visit_CallNode(self, node: CallNode):
        # print('call', node, node.fn, F.contained)
        if node.fn == F.contains().fn:
            # print('contains')
            assert len(node.params) == 2
            road = node.params[0]
            if not isinstance(road, LiteralNode):
                assert isinstance(road, CallNode), road
                assert road.fn == F.road_segment("TMP").fn, road
                assert len(road.params) == 1, road.params
                road = road.params[0]

            assert isinstance(road, LiteralNode), road
            assert isinstance(road.value, str), road
            return F.is_roadtype(road)
        # elif node.fn == F.contained().fn:
        #     # print('contains')
        #     assert len(node.params) == 2
        #     segment = node.params[1]
        #     if isinstance(segment, LiteralNode):
        #         assert isinstance(segment.value, str), (segment, node.fn)
        #         return F.is_roadtype(segment)
        #     else:
        #         assert isinstance(segment, CallNode), segment
        #         assert segment == F.same_region().fn, segment
        #         assert len(segment.params) == 1, segment.params
        #         assert isinstance(segment.params[0], LiteralNode), segment.params[0]
        #         assert isinstance(segment.params[0].value, str), segment.params[0]
        #         return F.is_roadtype(segment.params[0])
        return F.ignore_roadtype()

    def visit_TableNode(self, node: TableNode):
        return F.ignore_roadtype()

    def visit_ObjectTableNode(self, node: ObjectTableNode):
        return F.ignore_roadtype()

    def visit_CameraTableNode(self, node: CameraTableNode):
        return F.ignore_roadtype()

    def visit_CastNode(self, node: CastNode):
        return self(node.expr)


class PushInversionInForRoadTypePredicates(BaseTransformer):
    # def visit_ArrayNode(self, node: ArrayNode):
    #     raise Exception("Invalid Node Type")

    # def visit_CompOpNode(self, node: CompOpNode):
    #     raise Exception("Invalid Node Type")

    # def visit_BinOpNode(self, node: BinOpNode):
    #     raise Exception("Invalid Node Type")

    def visit_BoolOpNode(self, node: BoolOpNode):
        assert all(
            isinstance(e, (BoolOpNode, CallNode, UnaryOpNode)) for e in node.exprs
        ), node.exprs
        return super().visit_BoolOpNode(node)

    def visit_UnaryOpNode(self, node: UnaryOpNode):
        assert node.op == "invert"

        visited = super().visit_UnaryOpNode(node)
        assert isinstance(visited, UnaryOpNode), visited
        expr = visited.expr

        assert isinstance(expr, (BoolOpNode, CallNode)), expr
        # print('invert---')
        if isinstance(expr, BoolOpNode):
            if expr.op == "and":
                return self(BoolOpNode("or", [~e for e in expr.exprs]))
            else:
                return self(BoolOpNode("and", [~e for e in expr.exprs]))
        else:
            # print('invert', expr)
            assert expr.fn in [IS_ROADTYPE, IS_OTHER_ROADTYPE, IGNORE_ROADTYPE], expr.fn
            if expr.fn == IS_ROADTYPE:
                return F.is_other_roadtype(expr.params[0])
            elif expr.fn == IS_OTHER_ROADTYPE:
                return F.is_roadtype(expr.params[0])
            return expr

    # def visit_LiteralNode(self, node: LiteralNode):
    #     raise Exception("Invalid Node Type")

    # def visit_TableAttrNode(self, node: TableAttrNode):
    #     raise Exception("Invalid Node Type")

    def visit_CallNode(self, node: CallNode):
        assert node.fn in (IS_ROADTYPE, IS_OTHER_ROADTYPE, IGNORE_ROADTYPE), node.fn
        return node

    # def visit_TableNode(self, node: TableNode):
    #     raise Exception("Invalid Node Type")

    # def visit_ObjectTableNode(self, node: ObjectTableNode):
    #     raise Exception("Invalid Node Type")

    # def visit_CameraTableNode(self, node: CameraTableNode):
    #     raise Exception("Invalid Node Type")

    # def visit_CastNode(self, node: CastNode):
    #     raise Exception("Invalid Node Type")


class NormalizeInversionAndFlattenRoadTypePredicates(BaseTransformer):
    # def visit_ArrayNode(self, node: ArrayNode):
    #     raise Exception("Invalid Node Type")

    # def visit_CompOpNode(self, node: CompOpNode):
    #     raise Exception("Invalid Node Type")

    # def visit_BinOpNode(self, node: BinOpNode):
    #     raise Exception("Invalid Node Type")

    def visit_BoolOpNode(self, node: BoolOpNode):
        exprs = [self(e) for e in node.exprs]
        _exprs = []
        # Expand Ands
        for e in exprs:
            if isinstance(e, BoolOpNode) and e.op == node.op:
                _exprs.extend(e.exprs)
            else:
                _exprs.append(e)

        # Cleanup Ignore RoadType
        if node.op == "and":
            if all(isinstance(e, CallNode) and e.fn == IGNORE_ROADTYPE for e in _exprs):
                return F.ignore_roadtype()
            # Remove Ignore RoadType
            _exprs = [e for e in _exprs if not isinstance(e, CallNode) or e.fn != IGNORE_ROADTYPE]
        else:
            if any(isinstance(e, CallNode) and e.fn == IGNORE_ROADTYPE for e in exprs):
                return F.ignore_roadtype()

        # Boolean Absorption
        e_params = [e.params[0] for e in _exprs if isinstance(e, CallNode) and e.fn == IS_ROADTYPE]
        assert all(
            isinstance(e, LiteralNode) and isinstance(e.value, str) for e in e_params
        ), e_params
        _is_roadtypes = [
            v.lower()
            for v in (e.value for e in e_params if isinstance(e, LiteralNode))
            if isinstance(v, str)
        ]
        nested_exprs = [e for e in _exprs if isinstance(e, BoolOpNode) and e.op != node.op]
        assert len(_is_roadtypes) + len(nested_exprs) == len(_exprs), (
            len(_is_roadtypes),
            len(nested_exprs),
            len(_exprs),
            _exprs,
        )

        is_roadtypes: "set[str]" = set(_is_roadtypes)

        def is_absorbed(e: "CallNode | BoolOpNode"):
            if isinstance(e, CallNode):
                assert e.fn == IS_ROADTYPE, e.fn

                rt = e.params[0]
                assert isinstance(rt, LiteralNode), rt

                rt = rt.value.lower()
                return rt in is_roadtypes
            else:
                assert e.op == node.op, (node.op, e.op)
                all_roadtype = all(isinstance(e, CallNode) and e.fn == IS_ROADTYPE for e in e.exprs)
                if not all_roadtype:
                    return False
                exprs = [ee.params[0] for ee in e.exprs if isinstance(ee, CallNode)]
                assert all(
                    isinstance(e, LiteralNode) and isinstance(e.value, str) for e in exprs
                ), exprs
                return {
                    v.lower()
                    for v in (ee.value for ee in exprs if isinstance(ee, LiteralNode))
                    if isinstance(v, str)
                }.issubset(is_roadtypes)

        def all_not_absorbed(e: "BoolOpNode"):
            exprs = e.exprs
            assert all(isinstance(e, (CallNode, BoolOpNode)) for e in exprs), exprs
            exprs = (e for e in exprs if isinstance(e, (CallNode, BoolOpNode)))
            return not any(map(is_absorbed, exprs))

        nested_exprs = filter(all_not_absorbed, nested_exprs)

        _exprs = [*map(F.is_roadtype, sorted(is_roadtypes)), *nested_exprs]
        if len(_exprs) == 1:
            return _exprs[0]
        return BoolOpNode(node.op, _exprs)

    # def visit_UnaryOpNode(self, node: UnaryOpNode):
    #     raise Exception("Invalid Node Type")

    # def visit_LiteralNode(self, node: LiteralNode):
    #     raise Exception("Invalid Node Type")

    # def visit_TableAttrNode(self, node: TableAttrNode):
    #     raise Exception("Invalid Node Type")

    def visit_CallNode(self, node: CallNode):
        assert node.fn in (IS_ROADTYPE, IS_OTHER_ROADTYPE, IGNORE_ROADTYPE), node.fn
        if node.fn == IS_OTHER_ROADTYPE:
            rt = node.params[0]
            assert isinstance(rt, LiteralNode)

            rt_: "str" = rt.value.lower()
            assert rt_ != "road"
            return BoolOpNode("or", [F.is_roadtype(e) for e in OTHER_ROAD_TYPES[rt_]])
        return node

    # def visit_TableNode(self, node: TableNode):
    #     raise Exception("Invalid Node Type")

    # def visit_ObjectTableNode(self, node: ObjectTableNode):
    #     raise Exception("Invalid Node Type")

    # def visit_CameraTableNode(self, node: CameraTableNode):
    #     raise Exception("Invalid Node Type")

    # def visit_CastNode(self, node: CastNode):
    #     raise Exception("Invalid Node Type")


class FindRoadTypes(Visitor["set[str]"]):
    # def visit_ArrayNode(self, node: "ArrayNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_CompOpNode(self, node: "CompOpNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_BinOpNode(self, node: "BinOpNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    def visit_BoolOpNode(self, node: "BoolOpNode") -> "set[str]":
        return set.union(*map(self, node.exprs))

    # def visit_UnaryOpNode(self, node: "UnaryOpNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_LiteralNode(self, node: "LiteralNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_TableAttrNode(self, node: "TableAttrNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    def visit_CallNode(self, node: "CallNode") -> "set[str]":
        assert node.fn == IS_ROADTYPE, node.fn
        param = node.params[0]
        assert isinstance(param, LiteralNode), param
        assert isinstance(param.value, str), type(param.value)
        return {param.value.lower()}

    # def visit_TableNode(self, node: "TableNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_ObjectTableNode(self, node: "ObjectTableNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_CameraTableNode(self, node: "CameraTableNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")

    # def visit_CastNode(self, node: "CastNode") -> "set[str]":
    #     raise Exception("Invalid Node Type")


class InViewPredicate(Visitor[str]):
    def __init__(self, param_name: "str"):
        self.param_name = param_name

    # def visit_ArrayNode(self, node: "ArrayNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_CompOpNode(self, node: "CompOpNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_BinOpNode(self, node: "BinOpNode") -> "str":
    #     raise Exception("Invalid Node Type")

    def visit_BoolOpNode(self, node: "BoolOpNode") -> "str":
        return "(" + f" {node.op} ".join(map(self, node.exprs)) + ")"

    # def visit_UnaryOpNode(self, node: "UnaryOpNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_LiteralNode(self, node: "LiteralNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_TableAttrNode(self, node: "TableAttrNode") -> "str":
    #     raise Exception("Invalid Node Type")

    def visit_CallNode(self, node: "CallNode") -> "str":
        assert node.fn == IS_ROADTYPE, node.fn

        rt = node.params[0]
        assert isinstance(rt, LiteralNode), rt

        rt_: "str" = rt.value.lower()
        return f"('{rt_}' in {self.param_name})"
        # return rt in self.roadtypes

    # def visit_TableNode(self, node: "TableNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_ObjectTableNode(self, node: "ObjectTableNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_CameraTableNode(self, node: "CameraTableNode") -> "str":
    #     raise Exception("Invalid Node Type")

    # def visit_CastNode(self, node: "CastNode") -> "str":
    #     raise Exception("Invalid Node Type")


def create_inview_predicate(
    node: "PredicateNode",
) -> "tuple[list[str], str | bool]":
    node = KeepOnlyRoadTypePredicates()(node)
    # Note True/False will either disappear from all the predicates or propagate to the top
    if isinstance(node, LiteralNode):
        assert isinstance(node.value, bool), node.value
        return [], str(node.value)

    node = PushInversionInForRoadTypePredicates()(node)
    node = NormalizeInversionAndFlattenRoadTypePredicates()(node)
    # Note F.ignore_roadtype will either disappear from all the predicates or propagate to the top
    if isinstance(node, CallNode) and node.fn == IGNORE_ROADTYPE:
        return [], str(True)

    param_name = "roadtypes"
    predicate_str = InViewPredicate(param_name)(node)
    roadtypes = FindRoadTypes()(node)
    return sorted(list(roadtypes)), f"lambda {param_name}: {predicate_str}"
