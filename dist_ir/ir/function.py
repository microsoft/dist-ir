from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import copy
from typing import Any, Dict, Iterable, List, Tuple, Union

from .op import Op
from .value import Value


@dataclass(frozen=True)
class Function:
    """The core DistIR concept: a function.

    A function has a name, a list of input values, a list of operations, and a
    list of output values. Functions are immutable.
    """

    name: str
    ops: List[Op]
    inputs: List[Value]
    outputs: List[Value]

    # Map from Value -> List of Ops that consume it
    _consumers: Dict[Value, List[Op]] = field(init=False)

    def __post_init__(self):
        """Creates the _consumers map, verifies the function, and performs
        type inference. This is called automatically at initialization.
        """
        # Can't assign to frozen field:
        object.__setattr__(self, "_consumers", defaultdict(list))
        for op in self.ops:
            for in_edge in op.in_edges:
                self._consumers[in_edge].append(op)

        # Check that ops don't use values from the future
        self._verify_ops_in_topological_order()

        # Putting this import at the top level causes an import loop
        from ..executor.shape_inference import infer_shapes

        # TODO bring this back after refactor
        # infer_shapes(self)

    def _verify_ops_in_topological_order(self):
        seen = set()
        for inp in self.inputs:
            seen.add(inp)
        for op in self.ops:
            for in_edge in op.in_edges:
                if in_edge not in seen:
                    raise ValueError(
                        f"Ops are not in topological order: op {op.name} has "
                        f"unseen edge {in_edge}"
                    )
            for out_edge in op.out_edges:
                seen.add(out_edge)

    def get_consumers(self, value: Value) -> List[Op]:
        return self._consumers[value]

    def __str__(self):
        # TODO can we use the prettyprint output as __str__?
        return self.get_summary()

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

    def get_subfunction(self, op_names, name=None) -> Function:
        """Returns a Function comprised of the specified subset of ops."""
        # TODO I think we should use a set of ops instead of op_names if possible
        raise NotImplementedError("get_subfunction")
        subfunction = FunctionMaker(name)
        value_map = {}
        outputs = []
        op_names_set = set(op_names)
        for op_name in op_names:
            op = self.ops[op_name]
            subfunction_op_inputs = []
            for inp in op.in_edges:
                if inp not in value_map:
                    value_map[inp] = subfunction.add_input_value(inp.name, inp.type)
                subfunction_op_inputs.append(value_map[inp])
            output_names = [output.name for output in op.out_edges]
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
            for output in subfunction_op_outputs:
                # We need to explicitly set the subfunction outputs because some output
                # values might have consumers outside the subfunction (external).
                consumers = self._consumers[output]
                has_external_output = any([c not in op_names_set for c in consumers])
                if (
                    output in self.outputs or has_external_output
                ) and output.name not in op_names:
                    outputs.append(output)
                value_map[output] = output
        subfunction.set_outputs(outputs)
        return subfunction.finalize()


@dataclass
class FunctionMaker:
    """A helper class for creating Functions."""

    name: str = "foo"
    ops: List[Op] = field(default_factory=list)
    inputs: List[Value] = field(default_factory=list)
    outputs: List[Value] = field(default_factory=list)

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
          output_names: An optinal list of output value names.

        Returns:
          The outputs of the newly created op.
        """
        op = Op(
            op_type,
            name=name,
            in_edges=inputs,
            attributes=attributes,
            subfunctions=subfunctions,
            output_names=output_names,
        )
        self.ops.append(op)

        # Return the op outputs.
        num_out_edges = len(op.out_edges)
        if num_out_edges == 0:
            return None
        elif num_out_edges == 1:
            return op.out_edges[0]
        else:
            return tuple(op.out_edges)

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
            for in_edge in op.in_edges:
                is_output[in_edge] = False
            for out_edge in op.out_edges:
                is_output[out_edge] = True

        self.outputs = [v for v in is_output if is_output[v]]

    def _get_ops_in_topological_order_helper(self, name, visited, order):
        visited.add(name)

        out_edges = self.ops[name].out_edges
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

        return Function(self.name, self.ops, self.inputs, self.outputs)
