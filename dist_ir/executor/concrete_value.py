from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Dict, Tuple

from ..ir import Device, Op
from ..ir.type import Int64, Float32, Float64, Tensor


@dataclass(frozen=True)
class ConcreteValue:
    """A wrapper around a concrete value (e.g., an int, or a numpy.ndarray).
    The purpose of this wrapper is so that we can tag concrete values with
    device information when performing mixed interpretation in the simulator.
    """

    val: Any
    device: Device

    def __eq__(self, other):
        # Use numpy's array equality checking if val is an np.ndarray
        if isinstance(other, ConcreteValue):
            if isinstance(self.val, np.ndarray) and isinstance(other.val, np.ndarray):
                return self.device == other.device and (self.val == other.val).all()
                # TODO is there a better way to check np equality?
            else:
                return self.device == other.device and self.val == other.val
        return False

    def size(self):
        if isinstance(self.val, (np.ndarray, np.int64, np.float32, np.float64)):
            return self.val.size
        else:
            raise NotImplementedError()


def _wrap_concrete_implementation(implementation):
    """Wraps an implementation of an op that works on concrete values (e.g. numpy
    arrays) and returns an implementation that works on ConcreteValues.
    """

    def wrapped_implementation(op: Op, *args, **kwargs):
        # Unwrap arguments and find the device this op executes on
        device = None
        unwrapped_args = []
        for arg in args:
            assert isinstance(arg, ConcreteValue)
            if device is None:
                device = arg.device
            elif device is not None and device != arg.device:
                raise ValueError(
                    f"Op {op.op_type} received input values on multiple devices:"
                    f" {device} and {arg.device}"
                )
            unwrapped_args.append(arg.val)

        # Special case for constant (TODO better way?)
        if op.op_type == "Constant":
            device = op.attributes["device"]
        # assert device is not None

        outputs = implementation(op, *unwrapped_args, **kwargs)

        # Wrap outputs
        if isinstance(outputs, tuple):
            if len(op.outputs) > 1:
                return tuple(ConcreteValue(output, device) for output in outputs)
            else:
                # For ops like split that return a single tuple as output
                return ConcreteValue(tuple(output for output in outputs), device)
        else:
            return ConcreteValue(outputs, device)

    return wrapped_implementation


def wrap_concrete_register(register: Dict[Tuple[str, Tuple[type, ...]], Callable]):
    """Converts a concrete register (e.g., NumpyRegister) to one that runs on
    ConcreteValues. Note, this only works for single-device ops.

    `register`: a map: Tuple[OpType, Signature] -> Implementation.

    Returns a wrapped register of the same type, but operating on ConcreteValues.
    """
    wrapped_register = {
        (op_type, (ConcreteValue,) * len(signature)): _wrap_concrete_implementation(
            implementation
        )
        for (op_type, signature), implementation in register.items()
    }
    return wrapped_register
