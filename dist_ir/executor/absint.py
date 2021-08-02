"""
An abstract interpreter for DistIR programs. The abstract interpreter can be
instantiated to perform multiple analyses by providing it with a notion of
abstract state and semantics for each op type.

A semantics is a mapping: OpType -> List[Tuple[Signature+, Implementation]].
OpType is a string, Signature+ is a tuple of python types (e.g. Tensor,
np.ndarray) whose first element is the number of inputs, and Implementation is a
python function that takes the Op and the abstract state as input and modifies
the state in-place to reflect the execution of the op.

The order of implementations in the list is sorted by standard Python tuple order,
which is also most-precise-to-most-abstract order. E.g.:
    [
        ((1, Tensor), add_1_abs),
        ((2, np.ndarray, np.ndarray), add_conc),
        ((2, Tensor, Tensor), add_abs)
    ]

TODO also assume there are no entries with duplicate signatures?
"""

import numpy as np
from dist_ir.executor.concrete_value import ConcreteValue
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..ir import Function, Op, Value
from ..ir.type import Tensor, TupleType


def _abstract_type(concrete_type):
    if concrete_type == np.ndarray:
        return Tensor
    raise ValueError(f"Don't know how to abstract concrete type {concrete_type}")


def _abstractable_types(source_types: Sequence[type], target_types: Sequence[type]):
    """Returns true if each type in `source_types` is equal to or can be abstracted
    by the corresponding `target_type`.
    """
    if len(source_types) != len(target_types):
        return False
    for source_type, target_type in zip(source_types, target_types):
        if target_type != source_type and target_type != _abstract_type(source_type):
            return False
    return True


def update_semantics_with_register(
    semantics: Dict[str, List[Tuple[Tuple[type, ...], Callable]]],
    register: Dict[Tuple[str, Tuple[type, ...]], Callable],
):
    """Update `semantics` with the implementations in `register`. Can be used to
    build up a semantics for the AbstractInterpreter.

    `semantics`: a map: OpType -> List[Tuple[Signature+, Implementation]].
    See module docstring for more details.

    `register`: a map: Tuple[OpType, Signature] -> Implementation.
    """
    # TODO check duplicates?
    for (op_type, signature), implementation in register.items():
        implementations = semantics.get(op_type, [])
        implementations.append(((len(signature), *signature), implementation))
        semantics[op_type] = implementations
    # Sort all implementation lists
    for signature in semantics:
        semantics[signature].sort()


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

        `semantics`: Mapping: OpType -> List[Tuple[Signature+, Implementation]].
        See module docstring for more details.
        """
        self.AbstractState = AbstractState
        self.semantics = {} if semantics is None else semantics
        # TODO instead of passing the op, should we pass the attributes as kwargs?

    def interpret_pmap(self, op: Op, state: AbstractState):
        # TODO cache and reuse interpretation of subfunction if possible,
        # e.g., for the simulator

        # Find the op's inputs in state's environment
        inputs = tuple(state.env[v] for v in op.inputs)
        # Figure out if inputs are of type tuple or TupleType
        # (so that output can be made consistent)
        # TODO just use tuples instead of TupleType?
        assert isinstance(inputs[0], (tuple, TupleType))
        tuple_constructor = type(inputs[0])
        # Check that all the inputs to map over have the same length
        assert len(set(len(t) for t in inputs)) == 1
        # Zip the inputs so that we map over each corresponding value
        inputs = zip(*inputs)

        # Change state's function pointer to subfunction (TODO necessary?)
        function = state.function
        state.function = op.subfunctions[0]

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
        results = tuple(tuple_constructor(types) for types in zip(*results))
        # Put the results back into the state's environment
        for x, val in zip(op.outputs, results):
            state.env[x] = val

        state.function = function

        return state

    def interpret(
        self,
        function: Function,
        inputs: Sequence[Any],
        state: AbstractState = None,
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
                # Execute this op's semantics on the state
                inputs = (state.env[inp] for inp in op.inputs)
                implementation = _dispatch(self.semantics, op.op_type, inputs)
                implementation(op, state)

        return state


def _dispatch(
    semantics: Dict[str, List[Tuple[Tuple[type, ...], Callable]]],
    op_type: str,
    inputs: Sequence[Any],
) -> Callable:
    """Function dispatch. Looks at the types of `inputs` and finds the appropriate
    implementation function in `semantics`.

    `semantics`: Mapping: OpType -> List[Tuple[Signature+, Implementation]].
    See module docstring for more details.
    """
    implementations = semantics[op_type]
    input_types = tuple(
        type(input.val) if isinstance(input, ConcreteValue) else type(input)
        for input in inputs
    )

    # Find most precise implementation that matches input_types
    # (We break ties arbitrarily using lexicographic ordering)
    # Note: if this takes too long, memoize the answers
    # TODO do binary search?
    for (signature, implementation) in implementations:
        if signature[0] == len(input_types) and _abstractable_types(
            input_types, signature[1:]
        ):  # TODO signature -> (len, (types...))?
            # TODO continue: types. then create single mixed register
            return implementation

    raise ValueError(f"Could not dispatch {op_type} with input types {input_types}")


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
            if len(op.outputs) == 1:
                outputs = (outputs,)
            assert len(outputs) == len(op.outputs)
            for x, val in zip(op.outputs, outputs):
                state.env[x] = val

        return semantics

    return {signature: convert_impl(impl) for signature, impl in impls.items()}
