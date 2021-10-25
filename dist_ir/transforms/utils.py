from typing import Dict, Iterable, List

from ..ir import Function, Op


def get_op_to_stage_map(stages: Iterable[Function]) -> Dict[Op, Function]:
    """Given a list of stages, returns a map from individual op to
    the encompassing stage name."""
    op_to_stage = {}
    for stage in stages:
        for op in stage.ops:
            op_to_stage[op] = stage.name
    return op_to_stage


def get_stages_from_ops(
    op_to_stage: Dict[Op, Function], ops: Iterable[Op]
) -> List[Function]:
    """Given a list of ops and a map from op to encompassing stage,
    returns a list of encompassing stages."""
    seen = set()
    stages = []
    for op in ops:
        stage = op_to_stage[op]
        if stage not in seen:
            stages.append(stage)
    return stages
