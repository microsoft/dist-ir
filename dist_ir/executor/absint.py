from typing import Any, Dict, Sequence

from ..ir import Function, Op, Value
from ..ir.type import Tensor


class AbstractState:
    """An abstract state. env is an environment, i.e. a mapping from Value
    objects to abstract values.
    """

    def __init__(self, function: Function, inputs: Sequence[Any]):
        self.env: Dict[Value, Any] = dict(zip(function.inputs, inputs))
        self.function = function


class AbstractInterpreter:
    def __init__(self, AbstractState=AbstractState, semantics=None, Tuple=tuple):
        """An abstract interpreter: Given an abstract domain
        (abstract values and abstract implementation of all ops),
        this class provides methods to abstractly interpret a DistIR function on
        abstract values.

        `AbstractState`: subclass of absint.AbstractState to be used as abstract
        state.

        `semantics`: Mapping from (OpType, tuple of input types) -> Python function.
        Each function gets the Op and the abstract state as input and modifies
        the state in-place to reflect the execution of the op.

        `Tuple`: constructor for tuple values in the abstract domain. E.g.
        Python's tuple for the concrete domain, and TupleType for type domain.
        """
        self.AbstractState = AbstractState
        self.semantics = {} if semantics is None else semantics
        # TODO instead of passing the op, should we pass the attributes as kwargs?
        self.Tuple = Tuple

        # TODO some kind of type hierarchy for function call dispatch

    def interpret_pmap(self, op: Op, state: AbstractState):
        # TODO cache and reuse interpretation of subfunction if possible,
        # e.g., for the simulator

        # Find the op's inputs in state's environment
        inputs = tuple(state.env[v] for v in op.inputs)
        # Check that all the inputs to map over have the same length
        assert len(set(len(t) for t in inputs)) == 1
        # Zip the inputs so that we map over each corresponding value
        inputs = zip(*inputs)

        # Iterate over the inputs
        results = []
        for inps in inputs:
            # Interpret subfunction with appropriate inputs
            self.interpret(op.subfunctions[0], inps, state=state)
            # Find the outputs from the state's env
            outs = tuple(state.env[v] for v in op.subfunctions[0].outputs)
            results.append(outs)
            # TODO do we need to remove subfunction's values from env?

        # Unzip the results
        results = tuple(self.Tuple(types) for types in zip(*results))
        # Put the results back into the state's environment
        for x, val in zip(op.outputs, results):
            state.env[x] = val

        return state

    def interpret(
        self, function: Function, inputs: Sequence[Any], state: AbstractState = None
    ):
        """
        The result of the interpretation will be the final abstract state.
        From this, the abstract values output by the function can be extracted,
        but the state can also be used to build, e.g., a trace.
        """
        if state is None:
            state = self.AbstractState(function, inputs)
        else:
            state.env.update(zip(function.inputs, inputs))

        # Execute ops in topological order:
        for op in function.ops:
            if op.op_type == "Pmap":
                self.interpret_pmap(op, state)
            else:
                # Function dispatch:
                # I'm not sure whether to figure out input types and do function
                # dispatch here or in the wrapper that creates the semantics from
                # a symbol table, somthing like _convert_impls_to_semantics
                input_types = tuple(type(state.env[inp]) for inp in op.inputs)
                # Execute this op's semantics on the state
                self.semantics[op.op_type, input_types](op, state)

        return state


# The rest of this is file shows how to instantiate AbstractInterpreter to
# perform mixed concrete evaluation/type propagation
# TODO move this to another file instead

import numpy as np

# Each function gets the Op and input abstract values as input
# Each function returns a tuple of output abstract values


def _shape_abstract_to_concrete(op, x: Tensor):
    return np.array(x.shape, dtype=np.int64)


def _matmul_abstract(op, x, y):
    if not (x.dtype == y.dtype and x.device == y.device and x.shape[1] == y.shape[0]):
        raise Exception
        # _raise_type_error(op, x, y)
    return Tensor(dtype=x.dtype, shape=(x.shape[0], y.shape[1]), device=x.device)


def _slice_abstract_exact(op, x, starts, ends, axes):
    """The case when we know the slice indices concretely but x is abstract."""
    # TODO handle the other cases, e.g. negative indices
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(len(x.shape)))
    # Create a fake tensor and slice it because I'm lazy to work out the new shape
    y = np.zeros(x.shape)
    return Tensor(dtype=x.dtype, shape=y[slices].shape, device=x.device)


# TODO support variadic ops
MixedImplementations = {
    # ("Cast", (np.ndarray,)): _cast_concrete,
    # ("Min", (np.ndarray, np.ndarray)): _min_concrete,
    ("MatMul", (Tensor, Tensor)): _matmul_abstract,
    # ("Shape", (np.ndarray,)): _shape_concrete,
    ("Shape", (Tensor,)): _shape_abstract_to_concrete,
    # ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): _slice_concrete,
    ("Slice", (Tensor, np.ndarray, np.ndarray, np.ndarray)): _slice_abstract_exact,
}


def convert_impls_to_semantics(impls):
    """Converts a dictionary of semantics functions that take in input values
    and spit out output values to one that modifies an abstract state in place.
    """

    def convert_impl(impl_fn):
        def semantics(op: Op, state: AbstractState):
            # Find the op's inputs in state's environment
            inputs = (state.env[v] for v in op.inputs)
            # Execute the implementation on the inputs
            outputs = impl_fn(op, *inputs)
            # Put the outputs back into the state's environment
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for x, val in zip(op.outputs, outputs):
                state.env[x] = val

        return semantics

    return {signature: convert_impl(impl) for signature, impl in impls.items()}


MixedInterpreter = AbstractInterpreter(convert_impls_to_semantics(MixedImplementations))
