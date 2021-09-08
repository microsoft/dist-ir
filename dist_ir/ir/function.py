# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from collections import OrderedDict, defaultdict
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from frozendict import frozendict

from .op import Op
from .value import Value


@dataclass(frozen=True)
class Function:
    """The core DistIR concept: a function.

    A function has a name, a list of input values, a list of operations, and a
    list of output values. Functions are immutable.
    """

    name: str
    ops: Tuple[Op]
    inputs: Tuple[Value]
    outputs: Tuple[Value]

    # Map from Value -> List of Ops that consume it
    consumers: Dict[Value, Tuple[Op]] = field(init=False)
    # Map from Value -> Op that producers it
    producers: Dict[Value, Op] = field(init=False)

    def __post_init__(self):
        """Creates the consumers and producers maps, verifies the function,
        and performs type inference. This is called automatically at initialization.
        """
        consumers = defaultdict(list)
        for op in self.ops:
            for in_edge in op.inputs:
                consumers[in_edge].append(op)
            for out_edge in op.outputs:
                consumers[out_edge] = []
        for v in consumers:
            consumers[v] = tuple(consumers[v])
        producers = {}
        for op in self.ops:
            for out_edge in op.outputs:
                producers[out_edge] = op
        # Can't assign to frozen field:
        object.__setattr__(self, "consumers", frozendict(consumers))
        object.__setattr__(self, "producers", frozendict(producers))

        # Check that ops don't use values from the future
        self._verify_ops_in_topological_order()

    def _verify_ops_in_topological_order(self):
        seen = set()
        for inp in self.inputs:
            seen.add(inp)
        for op in self.ops:
            for in_edge in op.inputs:
                if in_edge not in seen:
                    raise ValueError(
                        f"Ops are not in topological order: op {op} has "
                        f"unseen edge {in_edge}"
                    )
            for out_edge in op.outputs:
                seen.add(out_edge)

    def __str__(self):
        # TODO can we use the prettyprint output as __str__?
        return self.get_summary()

    def last_use(self, value):
        """Returns the last op that uses the given value `value`."""
        return self.consumers[value][-1]

    def get_summary(self):
        output = ""
        output += "Function inputs:\n"
        for input_value in self.inputs:
            output += "  " + str(input_value) + "\n"
        output += "\n"
        output += "Function outputs:\n"
        for input_value in self.outputs:
            output += "  " + str(input_value) + "\n"
        output += "\n"
        output += "Ops:\n"
        for op in self.ops:
            output += str(op) + "\n"
        return output

    def has_input(self, value):
        """Checks whether the given value is an input of this function."""
        return value in self.inputs

    def has_output(self, value):
        """Checks whether the given value is an output of this function."""
        return value in self.outputs

    def get_subfunction(
        self, ops: List[Op], deepcopy: bool = False, name: Optional[str] = "bar"
    ) -> Function:
        """Returns a Function comprised of the specified subset of ops."""
        assert not deepcopy  # TODO We shouldn't need this functionality anymore
        subfunction = FunctionMaker(name)
        value_map = {}
        outputs = []
        for op in ops:
            subfunction_op_inputs = []
            for inp in op.inputs:
                if inp not in value_map:
                    if deepcopy:
                        value_map[inp] = subfunction.add_input_value(inp.name, inp.type)
                    else:
                        subfunction.inputs.append(inp)
                        value_map[inp] = inp
                subfunction_op_inputs.append(value_map[inp])
            output_names = [output.name for output in op.outputs]
            if deepcopy:
                subfunction_op_outputs = subfunction.add_op(
                    op.op_type,
                    name=op.name,
                    inputs=subfunction_op_inputs,
                    attributes=copy.deepcopy(op.attributes),
                    subfunctions=copy.deepcopy(op.subfunctions),
                    output_names=output_names,
                )
                if not isinstance(subfunction_op_outputs, tuple):
                    subfunction_op_outputs = (subfunction_op_outputs,)
            else:
                subfunction.ops.append(op)
                subfunction_op_outputs = op.outputs
            for orig_output, subfunction_output in zip(
                op.outputs, subfunction_op_outputs
            ):
                # We need to explicitly set the subfunction outputs because some output
                # values might have consumers outside the subfunction (external).
                has_external_output = False
                if (
                    orig_output in self.outputs
                    or len(set(self.consumers[orig_output]).difference(ops)) > 0
                ):
                    outputs.append(subfunction_output)
                value_map[orig_output] = subfunction_output
        subfunction.set_outputs(outputs)
        return subfunction.finalize()


@dataclass
class FunctionMaker:
    """A helper class for creating Functions."""

    name: str = "foo"
    ops: List[Op] = field(default_factory=list)
    inputs: List[Value] = field(default_factory=list)
    outputs: List[Value] = field(default_factory=list)

    # TODO wrong default values for Op
    def add_op(
        self,
        op_type,
        name=None,
        inputs: List[Value] = None,
        attributes: Dict[str, Any] = None,
        subfunctions: List["Function"] = None,
        output_names: List[str] = None,
    ) -> Union[None, Value, Tuple[Value, ...]]:
        """Adds an op to the function.

        Args:
          op_type: The op's type.
          name: The op's name.
          inputs: The input values for this op.
          attributes: Any op-specific attributes.
          subfunctions: Any subfunctions this op is wrapping.
          output_names: An optional list of output value names.

        Returns:
          The outputs of the newly created op.
        """
        op = Op(
            op_type,
            name=name,
            inputs=None if inputs is None else tuple(inputs),
            attributes=None if attributes is None else frozendict(attributes),
            subfunctions=() if subfunctions is None else tuple(subfunctions),
            output_names=None if output_names is None else tuple(output_names),
        )
        self.ops.append(op)

        # Return the op outputs.
        num_out_edges = len(op.outputs)
        if num_out_edges == 0:
            return None
        elif num_out_edges == 1:
            return op.outputs[0]
        else:
            return tuple(op.outputs)

    def add_input_value(self, name, value_type):
        """Adds an input value to the function and returns the value."""
        value = Value(name, value_type)
        if value in self.inputs:
            raise ValueError(f"Function already has input value {value}")
        self.inputs.append(value)
        return value

    def set_outputs(self, outputs: Iterable[Value]):
        """Sets the output of this function to be the given values. They must be
        valid values, i.e. outputs of some existing op in the function. This clears
        any previous outputs registered with this function.
        """
        self.outputs.clear()
        seen = set()
        for output in outputs:
            if output in seen:
                raise ValueError(f"Function already has output value {output}")
            seen.add(output)
            self.outputs.append(output)

    def set_outputs_auto(self):
        """Marks all sink nodes in the graph as output values."""
        is_output = OrderedDict()

        self.outputs.clear()
        for input_value in self.inputs:
            is_output[input_value] = True

        for op in self.ops:
            for in_edge in op.inputs:
                is_output[in_edge] = False
            for out_edge in op.outputs:
                if out_edge in is_output and not is_output[out_edge]:
                    print(
                        f"{out_edge.name} was not an output, but now is "
                        f"output of {op.op_type}"
                    )
                is_output[out_edge] = True

        self.outputs = [v for v in is_output if is_output[v]]

    def _get_ops_in_topological_order_helper(self, name, visited, order):
        visited.add(name)

        out_edges = self.ops[name].outputs
        for out_edge in out_edges:
            output_name = out_edge
            if output_name not in visited:
                self._get_ops_in_topological_order_helper(output_name, visited, order)

        order.append(name)

    def get_ops_in_topological_order(self):
        """Return ops in topological order. DEPRECATED, ops should always be
        topologically ordered.
        """
        visited = set()
        order = []
        for name in self.ops:
            if name not in visited:
                self._get_ops_in_topological_order_helper(name, visited, order)
        return order[::-1]

    def finalize(self) -> Function:
        """Returns the created Function. Outputs, if unspecified, are the sinks."""
        if len(self.outputs) == 0:
            self.set_outputs_auto()

        return Function(
            self.name, tuple(self.ops), tuple(self.inputs), tuple(self.outputs)
        )
