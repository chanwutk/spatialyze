from ...utils.F.common import default_location as dl, is_location_type
from ...predicate import GenSqlVisitor, TableNode, PredicateNode, call_node


@call_node
def distance(visitor: GenSqlVisitor, args: list[PredicateNode]):
    assert len(args) == 2, len(args)
    object1, object2 = args
    assert is_location_type(object1), repr(object1)
    assert is_location_type(object2), repr(object2)
    # assert isinstance(object1, TableNode), type(object1)
    # assert isinstance(object2, TableNode), type(object2)

    o1 = object1
    if isinstance(o1, TableNode):
        o1 = dl(o1)
    o2 = object2
    if isinstance(o2, TableNode):
        o2 = dl(o2)
    return (f"ST_Distance({visitor(o1)},{visitor(o2)})")
