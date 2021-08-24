"""
An abstract interpreter for DistIR programs. The abstract interpreter can be
instantiated to perform multiple analyses by providing it with a notion of
abstract state and semantics for each op type.

A semantics is a mapping: OpType -> List[Tuple[Signature, Implementation]].
OpType is a string, Signature is a tuple of python types (e.g. Tensor,
np.ndarray), and Implementation is a python function implementing the op that
additionally takes the Op as its first input and returns corresponding outputs.
(TODO instead of passing the op, should we pass the attributes as kwargs?)

The order of implementations in the list is sorted into groups according to
number of inputs, and the implementations in each group are sorted in
most-precise-to-most-abstract order. E.g.:
    [
        ((np.ndarray, np.ndarray), add_conc),
        ((Tensor, Tensor), add_abs),
        ((np.ndarray, np.ndarray, np.ndarray), add_3_conc),
    ]

TODO also assume there are no entries with duplicate signatures?
"""

import networkx as nx
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .concrete_value import ConcreteValue, wrap_concrete_register
from ..ir import Function, Op, Value
from ..ir.type import *
from .numpy_register import NumPyRegister
from .torch_register import TorchRegister
from .type_register import TypePropRegister
from .mixed_register import MixedRegister

# This is a graph of types supported by the AbstractInterpreter, with an edge
# (t1, t2) indicating that type t2 abstracts type t1.
# All values allowed by the AbstractInterpreter should have their types here.
_type_abstraction_graph: nx.DiGraph = nx.transitive_closure(
    nx.DiGraph(
        [
            (bool, Bool),
            (np.float32, Float32),
            (np.float64, Float64),
            (np.int32, Int32),
            (np.int64, Int64),
            (np.ndarray, Tensor),
            (torch.Tensor, Tensor),
            (tuple, TupleType),
            # TODO (if needed) have ConcreteBool, ConcreteFloat, etc
            (ConcreteValue, Bool),
            (ConcreteValue, Float32),
            (ConcreteValue, Float64),
            (ConcreteValue, Int32),
            (ConcreteValue, Int64),
            (ConcreteValue, Tensor),
            (ConcreteValue, Tensor),
            (ConcreteValue, TupleType),
        ]
    )
)

# The index of each type in the abstraction order
_type_index = {t: i for i, t in enumerate(nx.topological_sort(_type_abstraction_graph))}


def _abstracts(type1: type, type2: type):
    if type1 not in _type_abstraction_graph:
        raise ValueError(f"type1 ({type1}) not in type_abstraction_graph")
    if type2 not in _type_abstraction_graph:
        raise ValueError(f"type2 ({type2}) not in type_abstraction_graph")
    return type1 == type2 or _type_abstraction_graph.has_edge(type1, type2)


def _abstractable_types(source_types: Sequence[type], target_types: Sequence[type]):
    """Returns true if each type in `source_types` is equal to or can be abstracted
    by the corresponding `target_type`.
    """
    if len(source_types) != len(target_types):
        return False
    for source_type, target_type in zip(source_types, target_types):
        if not _abstracts(source_type, target_type):
            return False
    return True


def _signature_key(signature):
    """A key function to sort lists of signatures. See module docstring for
    details and example.
    """
    return (len(signature),) + tuple(_type_index[t] for t in signature)


def update_semantics_with_register(
    semantics: Dict[str, List[Tuple[Tuple[type, ...], Callable]]],
    register: Dict[Tuple[str, Tuple[type, ...]], Callable],
):
    """Update `semantics` with the implementations in `register`. Can be used to
    build up a semantics for the AbstractInterpreter.

    `semantics`: a map: OpType -> List[Tuple[Signature, Implementation]].
    See module docstring for more details.

    `register`: a map: Tuple[OpType, Signature] -> Implementation.
    """
    # TODO check duplicates?
    for (op_type, signature), implementation in register.items():
        implementations = semantics.get(op_type, [])
        implementations.append((signature, implementation))
        semantics[op_type] = implementations
    # Sort all implementation lists
    for op_type in semantics:
        semantics[op_type].sort(key=lambda x: _signature_key(x[0]))
    return semantics


def dispatch(
    semantics: Dict[str, List[Tuple[Tuple[type, ...], Callable]]],
    op_type: str,
    inputs: Sequence[Any],
) -> Callable:
    """Function dispatch. Looks at the types of `inputs` and finds the appropriate
    implementation function in `semantics`.

    `semantics`: Mapping: OpType -> List[Tuple[Signature, Implementation]].
    See module docstring for more details.
    """
    implementations = semantics[op_type]
    input_types = tuple(type(input) for input in inputs)

    # Find most precise implementation that matches input_types
    # (We break ties arbitrarily using lexicographic ordering)
    # Note: if this takes too long, memoize the answers
    # TODO do binary search?
    for (signature, implementation) in implementations:
        if _abstractable_types(input_types, signature):
            return implementation

    raise ValueError(f"Could not dispatch {op_type} with input types {input_types}")


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

        `semantics`: Mapping: OpType -> List[Tuple[Signature, Implementation]].
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
                # Find the op's inputs in state's environment
                inputs = tuple(state.env[v] for v in op.inputs)

                # Execute this op's semantics on the state
                implementation = dispatch(self.semantics, op.op_type, inputs)
                # TODO abstract inputs as necessary
                outputs = implementation(op, *inputs)

                # Put the outputs back into the state's environment
                if not isinstance(outputs, tuple):
                    assert len(op.outputs) == 1
                    outputs = (outputs,)
                assert len(outputs) == len(op.outputs)
                for x, val in zip(op.outputs, outputs):
                    state.env[x] = val

        return state


_semantics = {}
update_semantics_with_register(_semantics, TypePropRegister)
update_semantics_with_register(_semantics, wrap_concrete_register(NumPyRegister))
update_semantics_with_register(_semantics, wrap_concrete_register(TorchRegister))
update_semantics_with_register(_semantics, MixedRegister)
interpreter = AbstractInterpreter(AbstractState, _semantics)
