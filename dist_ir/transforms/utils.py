from typing import Dict, Iterable, List

from ..ir import Module


def get_op_to_stage_map(stages: Iterable[Module]) -> Dict[str, Module]:
    """Given a list of stages, returns a map from individual op name to
    encompassing stage."""
    op_to_stage = {}
    for stage in stages:
        for op_name in stage.get_ops():
            op_to_stage[op_name] = stage
    return op_to_stage


def get_stages_from_op_names(
    op_to_stage: Dict[str, Module], op_names: Iterable[str]
) -> List[Module]:
    """Given a list of op names and a map from op name to encompassing stage,
    returns a list of encompassing stages."""
    seen = set()
    stages = []
    for op_name in op_names:
        stage = op_to_stage[op_name]
        if stage not in seen:
            stages.append(stage)
    return stages
