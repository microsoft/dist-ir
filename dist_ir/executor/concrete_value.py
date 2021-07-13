from dataclasses import dataclass
import numpy as np
from typing import Any

from ..ir import Device


@dataclass(frozen=True)
class ConcreteValue:
    """A wrapper around a concrete value (e.g., an int, or a numpy.ndarray).
    The purpose of this wrapper is so that we can tag concrete values with
    device information when performing mixed interpretation in the simulator.
    """

    val: Any
    device: Device

    def size(self):
        if isinstance(self.val, np.ndarray):
            return self.val.size
        else:
            raise NotImplementedError()
