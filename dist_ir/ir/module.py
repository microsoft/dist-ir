from collections import OrderedDict, defaultdict
import copy
from typing import Any, Dict, Iterable, List, Tuple, Union

from .op import Op
from .value import Value


class Module:
    def __init__(self, name=None):
        self._ops = OrderedDict()
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()
        self._op_counter = defaultdict(int)
        self._consumers = defaultdict(list)
        self._name = name
        self._hash = None

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return self.get_summary()

    def __repr__(self):
        if self._name is not None:
            return self._name
        else:
            return object.__repr__(self)

    def __hash__(self):
        if self._hash is None:
            raise RuntimeError("Cannot hash unfinalized module!")
        return self._hash

    def __eq__(self, other):
        for op_name in self._ops:
            if op_name not in other._ops or self._ops[op_name] != other._ops[op_name]:
                return False
        for input_name in self._inputs:
            if (
                input_name not in other._inputs
                or self._inputs[input_name] != other._inputs[input_name]
            ):
                return False
        for output_name in self._outputs:
            if (
                output_name not in other._outputs
                or self._outputs[output_name] != other._outputs[output_name]
            ):
                return False
        return True

    def get_summary(self):
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

        # Update _consumers.
        out_edges = op.get_out_edges()
        for in_edge in inputs:
            self._consumers[in_edge.name].append(op.name)
        for out_edge in out_edges:
            self._consumers[out_edge.name] = []

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

    def get_consumers_for_value(self, name):
        return self._consumers[name]

    def set_outputs(self, outputs: Iterable[Value]):
        """Sets the output of this module to be the given values. They must be
        valid values, i.e. outputs of some existing op in the module. This clears
        any previous outputs registered with this module.
        """
        for output in outputs:
            # NOTE: Using consumers as a proxy for valid values
            if output.name not in self._consumers:
                raise ValueError(f"Module has no value {output}")
        self._outputs.clear()
        for output in outputs:
            if output.name in self._outputs:
                raise ValueError(
                    f"Module already has output value with name {output.name}"
                )
            self._outputs[output.name] = output

    def set_outputs_auto(self):
        """Marks all sink nodes in the graph as output values."""
        all_values = OrderedDict()
        is_output = OrderedDict()

        self._outputs.clear()
        for input_value_name, input_value in self._inputs.items():
            all_values[input_value_name] = input_value
            is_output[input_value_name] = True

        for op in self._ops.values():
            for in_edge in op.get_in_edges():
                is_output[in_edge.name] = False
            for out_edge in op.get_out_edges():
                all_values[out_edge.name] = out_edge
                is_output[out_edge.name] = True

        for output_value_name in is_output:
            if is_output[output_value_name]:
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

    def finalize(self):
        """Performs some standard verification and inference passes. Use at the
        end whenever creating a module. Assumes that the module will no longer be
        modified after this function is called.
        """
        # Putting this import at the top level causes an import loop
        from ..executor.shape_inference import infer_shapes

        self.verify_ops_in_topological_order()
        if len(self._outputs) == 0:
            self.set_outputs_auto()
        infer_shapes(self)
        self._hash = hash(tuple(self._ops.keys()))

    def get_submodule(self, op_names, name=None):
        """Returns a submodule comprised of the specified subset of ops."""
        submodule = Module(name)
        seen_values = set()
        outputs = []
        op_names_set = set(op_names)
        for op_name in op_names:
            op = self._ops[op_name]
            submodule_op = copy.deepcopy(op)
            submodule._ops[op_name] = submodule_op
            for in_edge in submodule_op.get_in_edges():
                if in_edge.name not in seen_values:
                    submodule._inputs[in_edge.name] = copy.deepcopy(in_edge)
                submodule._consumers[in_edge.name].append(op.name)
            for out_edge in submodule_op.get_out_edges():
                submodule._consumers[out_edge.name] = []
                seen_values.add(out_edge.name)
                external_output = out_edge.name in self._outputs or any(
                    [c not in op_names for c in self._consumers[out_edge.name]]
                )
                if external_output:
                    outputs.append(out_edge)
        submodule.set_outputs(outputs)
        submodule.finalize()
        return submodule
