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

    def size(self):
        if (
            isinstance(self.val, np.ndarray)
            or isinstance(self.val, np.int64)
            or isinstance(self.val, np.float32)
            or isinstance(self.val, np.float64)
        ):
            return self.val.size
        else:
            raise NotImplementedError()

    def to_abstract(self):
        def _resolve_dtype(dtype):
            if dtype == np.int64:
                return Int64()
            elif dtype == np.float32:
                return Float32()
            elif dtype == np.float64:
                return Float64()
            else:
                raise NotImplementedError(f"{dtype}")

        if isinstance(self.val, np.ndarray):
            return Tensor(
                shape=self.val.shape,
                dtype=_resolve_dtype(self.val.dtype),
                device=self.device,
            )
        elif isinstance(self.val, np.int64):
            return Int64(device=self.device)
        elif isinstance(self.val, np.float32):
            return Float32(device=self.device)
        elif isinstance(self.val, np.float64):
            return Float64(device=self.device)
        else:
            raise NotImplementedError(f"{type(self.val)}")


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
