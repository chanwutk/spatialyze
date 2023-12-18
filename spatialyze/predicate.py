from typing import Any, Callable, Generic, Literal, TypeVar, TypeGuard

BinOp = Literal["add", "sub", "mul", "div", "matmul"]
BoolOp = Literal["and", "or"]
CompOp = Literal["eq", "ne", "gt", "ge", "lt", "le"]
UnaryOp = Literal["invert", "neg"]


class PredicateNode:
    def __init__(self, *args, **kwargs):
        anns = self.__annotations__.keys()
        assert len(args) + len(kwargs) == len(
            anns
        ), f"Mismatch number of arguments: expecting {len(anns)}, received {len(args)} args and {len(kwargs)} kwargs"

        for k in kwargs:
            assert k in anns, f"{self.__class__.__name__} does not have attribute {k}"

        arg = iter(args)
        for k in anns:
            setattr(self, k, kwargs[k] if k in kwargs else next(arg))

    def __add__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "add", other)

    def __radd__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "add", self)

    def __sub__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "sub", other)

    def __rsub__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "sub", self)

    def __mul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "mul", other)

    def __rmul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "mul", self)

    def __truediv__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "div", other)

    def __rtruediv__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "div", self)

    def __matmul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(self, "matmul", other)

    def __rmatmul__(self, other):
        other = wrap_literal(other)
        return BinOpNode(other, "matmul", self)

    @staticmethod
    def __expand_exprs(op: "BoolOp", node: "PredicateNode") -> "list[PredicateNode]":
        if isinstance(node, BoolOpNode) and node.op == op:
            return node.exprs
        return [node]

    def __and__(self, other):
        other = wrap_literal(other)
        return BoolOpNode(
            "and",
            [
                *PredicateNode.__expand_exprs("and", self),
                *PredicateNode.__expand_exprs("and", other),
            ],
        )

    def __or__(self, other):
        other = wrap_literal(other)
        return BoolOpNode(
            "or",
            [
                *PredicateNode.__expand_exprs("or", self),
                *PredicateNode.__expand_exprs("or", other),
            ],
        )

    def __eq__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "eq", other)

    def __ne__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "ne", other)

    def __ge__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "ge", other)

    def __gt__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "gt", other)

    def __le__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "le", other)

    def __lt__(self, other):
        other = wrap_literal(other)
        return CompOpNode(self, "lt", other)

    def __invert__(self):
        return UnaryOpNode("invert", self)

    def __neg__(self):
        return UnaryOpNode("neg", self)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k).__repr__()}' for k in self.__annotations__)})"

    def __hash__(self):
        return id(self)


class ArrayNode(PredicateNode):
    exprs: "list[PredicateNode]"


def wrap_literal(x: Any) -> "PredicateNode":
    if isinstance(x, list):
        return arr(*x)
    if not isinstance(x, PredicateNode):
        return LiteralNode(x, True)
    return x


def arr(*exprs: "PredicateNode"):
    return ArrayNode([*map(wrap_literal, exprs)])


class CompOpNode(PredicateNode):
    left: "PredicateNode"
    op: CompOp
    right: "PredicateNode"


class BinOpNode(PredicateNode):
    left: "PredicateNode"
    op: BinOp
    right: "PredicateNode"


class BoolOpNode(PredicateNode):
    op: BoolOp
    exprs: "list[PredicateNode]"


class UnaryOpNode(PredicateNode):
    op: UnaryOp
    expr: "PredicateNode"


class LiteralNode(PredicateNode):
    value: Any
    python: bool


def lit(value: Any, python: bool = True):
    return LiteralNode(value, python)


class TableNode(PredicateNode):
    index: "int | None"

    def __getattr__(self, name: str) -> "TableAttrNode":
        return TableAttrNode(name, self, False)

    def __repr__(self):
        if self.index is None:
            return self.__class__.__name__
        return f"{self.__class__.__name__}[{self.index}]"


class ObjectTableNode(TableNode):
    index: int

    def __init__(self, index: int):
        self.index = index
        self.traj = TableAttrNode("trajCentroids", self, True)
        self.trans = TableAttrNode("translations", self, True)
        self.id = TableAttrNode("itemId", self, True)
        self.type = TableAttrNode("objectType", self, True)


class CameraTableNode(TableNode):
    def __init__(self, index: "int | None" = None):
        self.index = index
        self.time = TableAttrNode("timestamp", self, True)
        self.ego = TableAttrNode("egoTranslation", self, True)
        self.cam = TableAttrNode("cameraTranslation", self, True)


class TableAttrNode(PredicateNode):
    name: str
    table: "TableNode"
    shorten: bool


class ObjectTables:
    def __getitem__(self, index: int) -> "ObjectTableNode":
        return ObjectTableNode(index)


class CameraTables:
    def __getitem__(self, index: int) -> "CameraTableNode":
        return CameraTableNode(index)


objects = ObjectTables()
cameras = CameraTables()
camera = CameraTableNode()


FnKwarg = Callable[["GenSqlVisitor", list[PredicateNode], dict[str, PredicateNode]], str]
FnNoKwarg = Callable[["GenSqlVisitor", list[PredicateNode]], str]
FnOpt = FnKwarg | FnNoKwarg
# FnKwarg = Callable[["GenSqlVisitor", list[PredicateNode], dict[str, PredicateNode]], str]


def is_fn_kwarg(fn: "FnOpt") -> TypeGuard[FnKwarg]:
    return hasattr(fn, "__code__") and fn.__code__.co_argcount == 3


def is_fn_no_kwarg(fn: "FnOpt") -> TypeGuard[FnNoKwarg]:
    return hasattr(fn, "__code__") and fn.__code__.co_argcount == 2


def make_fn(fn_opt: FnOpt) -> FnKwarg:
    def fn(visitor: "GenSqlVisitor", args: "list[PredicateNode]", named_args: "dict[str, PredicateNode]"):
        if is_fn_kwarg(fn_opt):
            return fn_opt(visitor, args, named_args)
        elif is_fn_no_kwarg(fn_opt):
            return fn_opt(visitor, args)
        raise Exception('Function does not have the right signature')
    return fn


class CallNode(PredicateNode):
    _fn: "tuple[FnKwarg]"
    params: "list[PredicateNode]"

    def __init__(
        self,
        fn: "FnOpt",
        name: "str",
        params: "list[PredicateNode]",
        named_params: "dict[str, PredicateNode] | None" = None,
    ):
        assert fn.__code__.co_argcount in (2, 3), "Function does not have the right signature"
        if fn.__code__.co_argcount == 2:
            assert named_params is None, "Function does not have the right signature"
        self._fn = (make_fn(fn),)
        self.name = name
        self.params = params
        self.named_params = named_params or {}

    @property
    def fn(self) -> "FnKwarg":
        return self._fn[0]


def call_node(fn: "FnOpt"):
    def call_node_factory(
        *args: "PredicateNode | str | int | float | bool | list",
        **kargs: "PredicateNode | str | int | float | bool | list",
    ) -> "CallNode":
        return CallNode(
            fn,
            fn.__name__,
            [*map(wrap_literal, args)],
            {k: wrap_literal(v) for k, v in kargs.items()} if kargs else None,
        )

    return call_node_factory


class CastNode(PredicateNode):
    to: str
    expr: "PredicateNode"


def cast(expr: "Any", to: str) -> "CastNode":
    if not isinstance(expr, PredicateNode):
        expr = LiteralNode(expr, True)
    return CastNode(to, expr)


T = TypeVar("T")


class Visitor(Generic[T]):
    def __call__(self, node: "PredicateNode") -> T:
        attr = f"visit_{node.__class__.__name__}"
        assert hasattr(self, attr), "Unknown node type: " + node.__class__.__name__
        return getattr(self, attr)(node)

    def visit_ArrayNode(self, node: "ArrayNode") -> Any:
        for e in node.exprs:
            self(e)

    def visit_CompOpNode(self, node: "CompOpNode") -> Any:
        self(node.left)
        self(node.right)

    def visit_BinOpNode(self, node: "BinOpNode") -> Any:
        self(node.left)
        self(node.right)

    def visit_BoolOpNode(self, node: "BoolOpNode") -> Any:
        for e in node.exprs:
            self(e)

    def visit_UnaryOpNode(self, node: "UnaryOpNode") -> Any:
        self(node.expr)

    def visit_LiteralNode(self, node: "LiteralNode") -> Any:
        ...

    def visit_TableAttrNode(self, node: "TableAttrNode") -> Any:
        self(node.table)

    def visit_CallNode(self, node: "CallNode") -> Any:
        for p in node.params:
            self(p)

    def visit_TableNode(self, node: "TableNode") -> Any:
        ...

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> Any:
        ...

    def visit_CameraTableNode(self, node: "CameraTableNode") -> Any:
        ...

    def visit_CastNode(self, node: "CastNode") -> Any:
        self(node.expr)

    def reset(self):
        pass


class BaseTransformer(Visitor[PredicateNode]):
    def visit_ArrayNode(self, node: "ArrayNode") -> PredicateNode:
        return ArrayNode([self(e) for e in node.exprs])

    def visit_CompOpNode(self, node: "CompOpNode") -> PredicateNode:
        return CompOpNode(self(node.left), node.op, self(node.right))

    def visit_BinOpNode(self, node: "BinOpNode") -> PredicateNode:
        return BinOpNode(self(node.left), node.op, self(node.right))

    def visit_BoolOpNode(self, node: "BoolOpNode") -> PredicateNode:
        return BoolOpNode(node.op, [self(e) for e in node.exprs])

    def visit_UnaryOpNode(self, node: "UnaryOpNode") -> PredicateNode:
        return UnaryOpNode(node.op, self(node.expr))

    def visit_LiteralNode(self, node: "LiteralNode") -> PredicateNode:
        return node

    def visit_TableAttrNode(self, node: "TableAttrNode") -> PredicateNode:
        return TableAttrNode(node.name, self(node.table), node.shorten)

    def visit_CallNode(self, node: "CallNode") -> PredicateNode:
        return CallNode(node.fn, node.name, [self(p) for p in node.params], node.named_params)

    def visit_TableNode(self, node: "TableNode") -> PredicateNode:
        return node

    def visit_ObjectTableNode(self, node: "ObjectTableNode") -> PredicateNode:
        return node

    def visit_CameraTableNode(self, node: "CameraTableNode") -> PredicateNode:
        return node

    def visit_CastNode(self, node: "CastNode") -> PredicateNode:
        return CastNode(node.to, self(node.expr))


class ExpandBoolOpTransformer(BaseTransformer):
    def __call__(self, node: "PredicateNode"):
        if isinstance(node, BoolOpNode):
            exprs: "list[PredicateNode]" = []
            for expr in node.exprs:
                e = self(expr)
                if isinstance(e, BoolOpNode) and e.op == node.op:
                    exprs.extend(e.exprs)
                else:
                    exprs.append(e)
            return BoolOpNode(node.op, exprs)
        return super().__call__(node)


class FindAllTablesVisitor(Visitor["tuple[set[int], bool]"]):
    tables: "set[int]"
    camera: bool

    def __init__(self):
        self.tables = set()
        self.camera = False

    def __call__(self, node: "PredicateNode"):
        super().__call__(node)
        return self.tables, self.camera

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        self.tables.add(node.index)

    def visit_CameraTableNode(self, node: "CameraTableNode"):
        self.camera = True

    def reset(self):
        self.tables = set()
        self.camera = False


class MapTablesTransformer(BaseTransformer):
    mapping: "dict[int, int]"

    def __init__(self, mapping: "dict[int, int]"):
        self.mapping = mapping

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        if node.index in self.mapping:
            return objects[self.mapping[node.index]]
        return node


class NormalizeArrayAtTime(BaseTransformer):
    def visit_BinOpNode(self, node: "BinOpNode"):
        if node.op == "matmul":
            left = node.left
            if isinstance(left, ArrayNode):
                return self(
                    ArrayNode([BinOpNode(expr, node.op, node.right) for expr in left.exprs])
                )
        return node


normalizers: "list[BaseTransformer]" = [ExpandBoolOpTransformer(), NormalizeArrayAtTime()]


def normalize(predicate: "PredicateNode") -> "PredicateNode":
    for normalizer in normalizers:
        normalizer.reset()
        predicate = normalizer(predicate)
    return predicate


BIN_OP: "dict[BinOp, str]" = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

BOOL_OP: "dict[BoolOp, str]" = {
    "and": " AND ",
    "or": " OR ",
}

COMP_OP: "dict[CompOp, str]" = {
    "eq": "=",
    "ne": "<>",
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
}

UNARY_OP: "dict[UnaryOp, str]" = {
    "invert": "NOT ",
    "neg": "-",
}


class GenSqlVisitor(Visitor[str]):
    # Needed to that we ony add `@camera.time` once
    prev_timed: "set[PredicateNode]" = set()

    def visit_ArrayNode(self, node: "ArrayNode"):
        elts = ",".join(self(e)[5 if isinstance(e, ArrayNode) else 0 :] for e in node.exprs)
        return f"ARRAY[{elts}]"

    def visit_BinOpNode(self, node: "BinOpNode"):
        if node.op == "matmul":
            self.prev_timed.add(node.left)

        left = self(node.left)
        right = self(node.right)
        if node.op != "matmul":
            return f"({left}{BIN_OP[node.op]}{right})"

        if isinstance(node.left, TableAttrNode) and node.left.name == "bbox":
            return f"objectBBox({self(node.left.table.id)},{right})"
        if "Headings" in left:
            return f"headingAtTimestamp({left},{right})"
        return f"valueAtTimestamp({left},{right})"

    def visit_BoolOpNode(self, node: "BoolOpNode"):
        op = BOOL_OP[node.op]
        return f"({op.join(self(e) for e in node.exprs)})"

    def visit_CallNode(self, node: "CallNode"):
        fn = node.fn
        return fn(self, node.params, node.named_params)

    def visit_TableAttrNode(self, node: "TableAttrNode"):
        table = node.table
        assert isinstance(table, (ObjectTableNode, CameraTableNode)), "table type not supported"

        if isinstance(table, ObjectTableNode):
            if node.name in IS_TEMPORAL:
                raise Exception('Dropping support for temporal attributes -> use object')
            if node.name in IS_TEMPORAL and IS_TEMPORAL[node.name] and node not in self.prev_timed:
                self.prev_timed.add(node)
                return self(node @ camera.time)
            return resolve_object_attr(self, node.name, table.index)
        elif isinstance(table, CameraTableNode):
            return resolve_camera_attr(self, node.name, table.index)

    def visit_TableNode(self, node: "TableNode"):
        if isinstance(node, ObjectTableNode):
            return f"valueAtTimestamp({resolve_object_attr(self, 'translations', node.index)},timestamp)"
        elif isinstance(node, CameraTableNode):
            return resolve_camera_attr(self, 'cameraTranslation', node.index)

    def visit_CompOpNode(self, node: "CompOpNode"):
        left = self(node.left)
        right = self(node.right)
        return f"({left}{COMP_OP[node.op]}{right})"

    def visit_LiteralNode(self, node: "LiteralNode"):
        value = node.value
        if isinstance(value, str) and node.python:
            return f"'{value}'"
        else:
            return str(value)

    def visit_UnaryOpNode(self, node: "UnaryOpNode"):
        return f"({UNARY_OP[node.op]}{self(node.expr)})"

    def visit_ObjectTableNode(self, node: "ObjectTableNode"):
        return self(node.traj)

    def visit_CameraTableNode(self, node: "CameraTableNode"):
        return self(node.cam)

    def visit_CastNode(self, node: "CastNode"):
        return f"({self(node.expr)})::{node.to}"


def resolve_object_attr(visitior: "Visitor", attr: str, num: "int | None" = None):
    if num is None:
        return attr
    return f"t{num}.{attr}"


def resolve_camera_attr(visitior: "Visitor", attr: str, num: "int | None" = None):
    if num is None:
        return attr
    return f"c{num}.{attr}"


IS_TEMPORAL: "dict[str, bool]" = {
    "itemId": False,
    "cameraId": False,
    "objectType": False,
    "trajCentroids": True,
    "translations": True,
    "itemHeadings": True,
    "bbox": True,
}

# TODO: this is duplicate with the one in database.py
TRAJECTORY_COLUMNS: "list[tuple[str, str]]" = [
    ("itemId", "TEXT"),
    ("cameraId", "TEXT"),
    ("objectType", "TEXT"),
    # ("roadTypes", "ttext"),
    ("trajCentroids", "tgeompoint"),
    ("translations", "tgeompoint"),  # [(x,y,z)@today, (x2, y2,z2)@tomorrow, (x2, y2,z2)@nextweek]
    ("itemHeadings", "tfloat"),
    # ("color", "TEXT"),
    # ("largestBbox", "STBOX")
    # ("roadPolygons", "tgeompoint"),
    # ("period", "period") [today, nextweek]
]


CAMERA_COLUMNS: "list[tuple[str, str]]" = [
    ("cameraId", "TEXT"),
    ("frameId", "TEXT"),
    ("frameNum", "Int"),
    ("fileName", "TEXT"),
    ("cameraTranslation", "geometry"),
    ("cameraRotation", "real[4]"),
    ("cameraIntrinsic", "real[3][3]"),
    ("egoTranslation", "geometry"),
    ("egoRotation", "real[4]"),
    ("timestamp", "timestamptz"),
    ("cameraHeading", "real"),
    ("egoHeading", "real"),
]
