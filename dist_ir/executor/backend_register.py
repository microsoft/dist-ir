# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .numpy_register import NumPyRegister
from .torch_register import TorchRegister

BackendRegister = {"numpy": NumPyRegister, "torch": TorchRegister}
