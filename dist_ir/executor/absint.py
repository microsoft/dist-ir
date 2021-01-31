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
    def __init__(self, AbstractState=AbstractState, semantics=None):
        """An abstract interpreter: Given an abstract domain
        (abstract values and abstract implementation of all ops),
        this class provides methods to abstractly interpret a DistIR function on
        abstract values.

        `AbstractState`: subclass of absint.AbstractState to be used as abstract
        state.

        `semantics`: Mapping from (OpType, tuple of input types) -> Python function.
        Each function gets the Op and the abstract state as input and modifies
        the state in-place to reflect the execution of the op.
        """
        self.AbstractState = AbstractState
        self.semantics = {} if semantics is None else semantics

        # TODO some kind of type hierarchy for function call dispatch

    def interpret(self, function: Function, inputs: Sequence[Any]):
        """
        The result of the interpretation will be the final abstract state.
        From this, the abstract values output by the function can be extracted,
        but the state can also be used to build, e.g., a trace.
        """
        state = self.AbstractState(function, inputs)

        # Execute ops in topological order:
        for op in function.ops:
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

# TODO instead of passing the op, should we pass the attributes as kwargs?
def _cast_concrete(op, x):
    proto_dtype = op.attributes["to"]
    dtype = {
        1: np.float32,
        6: np.int32,
        7: np.int64,
        9: np.bool,
    }[proto_dtype]
    return x.astype(dtype)


def _shape_concrete(op, x):
    return np.array(x.shape, dtype=np.int64)


def _shape_abstract_to_concrete(op, x: Tensor):
    return np.array(x.shape, dtype=np.int64)


def _min_concrete(op, x, y):
    return np.minimum(x, y)


def _matmul_abstract(op, x, y):
    if not (x.dtype == y.dtype and x.device == y.device and x.shape[1] == y.shape[0]):
        raise Exception
        # _raise_type_error(op, x, y)
    return Tensor(dtype=x.dtype, shape=(x.shape[0], y.shape[1]), device=x.device)


def _slice_concrete(op, x, starts, ends, axes):
    # TODO handle the other cases, e.g. negative indices
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(x.ndim))
    return x[slices]


def _slice_abstract_exact(op, x, starts, ends, axes):
    """The case when we know the slice indices concretely but x is abstract."""
    # TODO handle the other cases, e.g. negative indices
    slices = {axis: slice(s, e) for (s, e, axis) in zip(starts, ends, axes)}
    slices = tuple(slices.get(d, slice(None)) for d in range(len(x.shape)))
    # Create a fake tensor and slice it because I'm lazy to work out the new shape
    y = np.zeros(x.shape)
    return Tensor(dtype=x.dtype, shape=y[slices].shape, device=x.device)


def _slice_abstract(op, x, starts, ends, axes):
    # We don't know the shape of the output, so:
    return Tensor(dtype=x.dtype, shape=None, device=x.device)


# TODO support variadic ops
MixedImplementations = {
    ("Cast", (np.ndarray,)): _cast_concrete,
    ("Min", (np.ndarray, np.ndarray)): _min_concrete,
    ("MatMul", (Tensor, Tensor)): _matmul_abstract,
    ("Shape", (np.ndarray,)): _shape_concrete,
    ("Shape", (Tensor,)): _shape_abstract_to_concrete,
    ("Slice", (np.ndarray, np.ndarray, np.ndarray, np.ndarray)): _slice_concrete,
    ("Slice", (Tensor, np.ndarray, np.ndarray, np.ndarray)): _slice_abstract_exact,
    ("Slice", (Tensor, Tensor, Tensor, Tensor)): _slice_abstract,
}


def _convert_impls_to_semantics(impls):
    """Converts a dictionary of semantics functions that take in input values
    and spit out output values to one that modifies an abstract state in place.
    """

    def convert_impl(impl_fn):
        def semantics(op: Op, state: AbstractState):
            # Find the op's inputs in state's environment
            inputs = (state.env[v] for v in op.inputs)
            outputs = impl_fn(op, *inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for x, val in zip(op.outputs, outputs):
                state.env[x] = val

        return semantics

    return {signature: convert_impl(impl) for signature, impl in impls.items()}


MixedInterpreter = AbstractInterpreter(
    _convert_impls_to_semantics(MixedImplementations)
)
