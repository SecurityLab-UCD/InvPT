"""Transformation interface for Java programs"""

from java_transform.utils import spat_caller
from java_transform.utils import Program
from modeling.dataloader import AugType

TRANSFORMATION_MAP = {
    AugType.LOCALVARRENAMING: spat_caller(0),
    AugType.FOR2WHILE: spat_caller(1),
    AugType.WHILE2FOR: spat_caller(2),
    AugType.REVERSEIFELSE: spat_caller(3),
    AugType.PP2ADDASSIGNMENT: spat_caller(6),
    AugType.ADDASSIGNMENT2EQUALASSIGNMENT: spat_caller(7),
}


def augment_accumulatively(ps: list[Program]) -> list[Program]:
    ps_map = {i: p for i, p in ps}
    for aug_type in TRANSFORMATION_MAP.keys():
        ps = TRANSFORMATION_MAP[aug_type](ps)
        # if for all program in result, replace the original program in ps with the same id
        for i, p in ps:
            ps_map[i] = p

    # convert the map back to list
    return [(i, p) for i, p in ps_map.items()]
