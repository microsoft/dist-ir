from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, List, Tuple

from frozendict import frozendict

from .op_register import OpRegister
from .value import Value
from .type import Type


@dataclass(frozen=True)
class Op:
    op_type: str
    name: str = ""
    inputs: Tuple[Value] = field(default_factory=tuple)
    attributes: Dict[str, Any] = field(default_factory=frozendict)
    subfunctions: Tuple["Function"] = field(default_factory=tuple)
    outputs: Tuple[Value] = field(init=False)

    # These are not fields, just parameters to init and post_init:
    output_names: InitVar[Tuple[str]] = None
    output_types: InitVar[Tuple[Type]] = None
    output_values: InitVar[Tuple[Type]] = None

    def __post_init__(self, output_names, output_types, output_values):
        if self.op_type == "Pmap":
            # Handle pmap specially
            assert len(self.subfunctions) == 1
            # Number of inputs is arbitrary but positive
            assert len(self.inputs) > 0
            # Number of inputs matches subfunction
            assert len(self.inputs) == len(self.subfunctions[0].inputs)
            # Number of outputs is given by subfunction
            num_outputs = len(self.subfunctions[0].outputs)
            assert output_values is None or len(output_values) == num_outputs
        else:
            if self.op_type not in OpRegister:
                raise ValueError(f"Invalid op type {self.op_type}")
            # Check that we got the right number of inputs
            if not OpRegister[self.op_type].variadic_inputs:
                num_input_types = OpRegister[self.op_type].num_inputs
                if len(self.inputs) != num_input_types:
                    raise ValueError(
                        f"Op {self.name} ({self.op_type}) has {len(self.inputs)} inputs; "
                        f"{num_input_types} expected"
                    )
            # Number of outputs is given by OpRegister
            variadic_outputs = OpRegister[self.op_type].variadic_outputs
            if variadic_outputs:
                if output_names is None:
                    raise ValueError(
                        f"Op {self.name} ({self.op_type}) has variadic "
                        f"outputs, so output names must be specified"
                    )
                num_outputs = len(output_names)
            else:
                num_outputs = OpRegister[self.op_type].num_outputs

        if output_values is not None:
            object.__setattr__(
                self, "outputs", output_values
            )  # Can't assign to frozen field
            return

        # Create the correct number of output values with appropriate types
        # if self.outputs is None:
        if output_names is None:
            output_names = [f"{self.name}_out_{i}" for i in range(num_outputs)]
        elif len(output_names) != num_outputs:
            raise ValueError(
                f"Op {self.name} ({self.op_type}) has {len(output_names)} outputs; "
                f"{num_outputs} expected"
            )
        if output_types is None:
            output_types = [None for i in range(num_outputs)]
        elif len(output_types) != num_outputs:
            raise ValueError(
                f"Op {self.name} ({self.op_type}) has {len(output_types)} outputs; "
                f"{num_outputs} expected"
            )
        outputs = tuple(
            Value(out_name, out_type)
            for out_name, out_type in zip(output_names, output_types)
        )
        object.__setattr__(self, "outputs", outputs)  # Can't assign to frozen field
