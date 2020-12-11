from collections import OrderedDict


class ModuleView:
    def __init__(self, module, ops, name=None):
        self._ops = OrderedDict()
        for op in ops:
            self._ops[op.name] = ops
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()

        seen_values = set()
        for op in ops:
            inputs = op.get_in_edges()
            outputs = op.get_out_edges()
            for input in inputs:
                if input.name not in seen_values:
                    self._inputs[input.name] = input
            for output in outputs:
                seen_values.add(output.name)
                consumers = module.get_consumers_for_value(output.name)
                has_external_consumer = module.is_output(output.name) or any(
                    [c not in self._ops for c in consumers]
                )
                if has_external_consumer:
                    self._outputs[output.name] = output

        self._name = name
        self._hash = hash(tuple(self._ops.keys()))

    def __str__(self):
        if self._name is not None:
            return self._name
        output = ""
        for i, op in enumerate(self._ops.values()):
            if i == len(self._ops) - 1:
                output += str(op)
            else:
                output += str(op) + "\n"
        return output

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._ops == other._ops

    def get_ops(self):
        return self._ops

    def get_inputs(self):
        return self._inputs.values()

    def get_outputs(self):
        return self._outputs.values()
