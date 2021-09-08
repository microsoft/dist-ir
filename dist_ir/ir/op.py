# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    output_values: InitVar[Tuple[Value]] = None

    def __post_init__(self, output_names, output_types, output_values):
        # Check output_{names,types,values} have same length
        given_num_outputs = set(
            len(x) for x in [output_names, output_types, output_values] if x is not None
        )
        if len(given_num_outputs) == 0:
            given_num_outputs = None
        elif len(given_num_outputs) == 1:
            given_num_outputs = list(given_num_outputs)[0]
        else:
            raise ValueError(
                "output_names, output_types, and output_values (if provided) "
                "must have same length. Got:\n"
                f"{output_names}\n{output_types}\n{output_values}"
            )

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
            if OpRegister[self.op_type].variadic_outputs:
                if given_num_outputs is None:
                    raise ValueError(
                        f"Op {self.name} ({self.op_type}) has variadic "
                        "outputs, so one of output_names or output_values must "
                        "be specified"
                    )
                num_outputs = given_num_outputs
            else:
                num_outputs = OpRegister[self.op_type].num_outputs
                assert given_num_outputs is None or num_outputs == given_num_outputs

        if output_values is None:
            # Create the correct number of output values with appropriate types
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
            output_values = tuple(
                Value(out_name, out_type)
                for out_name, out_type in zip(output_names, output_types)
            )
        object.__setattr__(
            self, "outputs", output_values
        )  # Can't assign to frozen field
