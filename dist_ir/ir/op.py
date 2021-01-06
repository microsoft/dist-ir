from dataclasses import dataclass, field, InitVar
from frozendict import frozendict
from typing import Any, Dict, List, Tuple

from .op_register import OpRegister
from .value import Value


@dataclass(frozen=True)
class Op:
    op_type: str
    name: str = ""
    in_edges: Tuple[Value] = field(default_factory=tuple)
    attributes: Dict[str, Any] = field(
        default_factory=frozendict
    )  # TODO: Replace with frozendict
    subfunctions: Tuple["Function"] = field(default_factory=tuple)
    out_edges: Tuple[Value] = field(init=False)

    # This is not a field, just a parameter to init and post_init:
    output_names: InitVar[Tuple[str]] = None

    def __post_init__(self, output_names):
        if self.op_type == "Pmap":
            # Handle pmap specially
            assert len(self.subfunctions) == 1
            # Number of inputs is arbitrary but positive
            assert len(self.in_edges) > 0
            # Number of inputs matches subfunction
            assert len(self.in_edges) == len(self.subfunctions[0].inputs)
            # Number of outputs is given by subfunction
            num_outputs = len(self.subfunctions[0].outputs)

        else:
            if self.op_type not in OpRegister:
                raise ValueError(f"Invalid op type {self.op_type}")
            # Check that we got the right number of inputs
            assert len(self.in_edges) == OpRegister[self.op_type].num_inputs
            # Number of outputs is given by OpRegister
            num_outputs = OpRegister[self.op_type].num_outputs

        # Create the correct number of output values with type=None
        if output_names is None:
            output_names = [f"{self.name}_out_{i}" for i in range(num_outputs)]
        else:
            assert len(output_names) == num_outputs
        out_edges = tuple(Value(out_name, None) for out_name in output_names)
        object.__setattr__(self, "out_edges", out_edges)  # Can't assign to frozen field
