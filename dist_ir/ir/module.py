from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, Union

from .op import Op
from .value import Value


class Module:
    def __init__(self):
        self._ops = OrderedDict()
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()
        self._op_counter = defaultdict(int)
        self._consumers = defaultdict(int)

    def __str__(self):
        output = ""
        output += "Module inputs:\n"
        for input_value in self._inputs.values():
            output += "  " + str(input_value) + "\n"
        output += "\n"
        output += "Module outputs:\n"
        for input_value in self._outputs.values():
            output += "  " + str(input_value) + "\n"
        output += "\n"
        output += "Ops:\n"
        for op in self._ops.values():
            output += str(op) + "\n"
        return output

    # TODO: Convert to property
    def get_ops(self):
        """Returns all ops in the module."""
        return self._ops

    def is_op(self, name):
        """Checks whether a op exists with the specified name."""
        return name in self._ops

    def is_input(self, name):
        """Checks whether an input value exists with the specified name."""
        return name in self._inputs

    def is_output(self, name):
        """Checks whether an output value exists with the specified name."""
        return name in self._outputs

    def get_op(self, name):
        """Returns the op with the specified name if it exists."""
        if name not in self._ops:
            return None
        return self._ops[name]

    def get_input(self, name):
        """Returns the input value with the specified name if it exists."""
        if name not in self._inputs:
            return None
        return self._inputs[name]

    def get_inputs(self):
        """Returns the module inputs."""
        return self._inputs.values()

    def get_outputs(self):
        """Returns the module outputs."""
        return self._outputs.values()

    def add_op(
        self,
        op_type,
        name=None,
        inputs: List[Value] = None,
        attributes: Dict[str, Any] = None,
        submodules: List["Module"] = None,
        metadata: Dict[str, Any] = None,
        output_names: List[str] = None,
    ) -> Union[None, Value, Tuple[Value, ...]]:
        """Adds an op to the graph.

        Args:
          op_type: The op's type.
          name: The op's name.
          inputs: The input values for this op.
          attributes: Any op-specific attributes.
          submodules: Any submodules this op is wrapping.
          metadata: Any op-specific metadata.
          output_names: An optinal list of output value names.

        Returns:
          The outputs of the newly created op.
        """
        if name in self._ops:
            raise ValueError(f"op with name {name} already exists!")
        elif name is None or name == "":
            name = f"{op_type}_#{self._op_counter[op_type]}"
        op = Op(
            name,
            op_type,
            in_edges=inputs,
            attributes=attributes,
            submodules=submodules,
            metadata=metadata,
            output_names=output_names,
        )
        self._ops[name] = op
        self._op_counter[op_type] += 1

        # Update the module outputs.
        out_edges = op.get_out_edges()
        for out_edge in out_edges:
            if out_edge.name in self._outputs:
                raise ValueError(
                    f"Module already has output value with name {out_edge.name}"
                )
            self._outputs[out_edge.name] = out_edge
            self._consumers[out_edge.name] = 0
        for in_edge in inputs:
            if in_edge.name in self._outputs:
                self._consumers[in_edge.name] += 1
                del self._outputs[in_edge.name]

        # Return the op outputs.
        num_out_edges = len(out_edges)
        if num_out_edges == 0:
            return None
        elif num_out_edges == 1:
            return out_edges[0]
        else:
            return tuple(out_edges)

    def add_input_value(self, name, value_type):
        """Adds an input value to the graph and returns the value."""
        value = Value(name=name, value_type=value_type)
        if value.name in self._inputs:
            raise ValueError(f"Module already has input value with name {value.name}")
        self._inputs[value.name] = value
        return value

    def get_consumers_for_out_edge(self, name):
        return self._consumers[name]

    def find_output_values(self):
        """Marks all sink nodes in the graph as output values."""
        all_values = {}
        consumed_values = {}

        for input_value_name, input_value in self._inputs.items():
            all_values[input_value_name] = input_value

        for op in self._ops.values():
            for in_edge in op.get_in_edges():
                consumed_values[in_edge.name] = in_edge
            for out_edge in op.get_out_edges():
                all_values[out_edge.name] = out_edge

        output_value_names = set(all_values.keys()).difference(
            set(consumed_values.keys())
        )
        for output_value_name in output_value_names:
            self._outputs[output_value_name] = all_values[output_value_name]

    def _get_ops_in_topological_order_helper(self, name, visited, order):
        visited.add(name)

        out_edges = self._ops[name].get_out_edges()
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
        for name in self._ops:
            if name not in visited:
                self._get_ops_in_topological_order_helper(name, visited, order)
        return order[::-1]

    def verify_ops_in_topological_order(self):
        seen = set()
        for input_name in self._inputs:
            seen.add(input_name)

        for op_name, op in self._ops.items():
            for in_edge in op.get_in_edges():
                if in_edge.name not in seen:
                    raise ValueError(
                        f"Ops are not in topological order: op {op_name} has "
                        f"unseen edge {in_edge}"
                    )
            for out_edge in op.get_out_edges():
                seen.add(out_edge.name)
