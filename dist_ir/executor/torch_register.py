# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


TorchRegister = {
    ("Add", (torch.Tensor, torch.Tensor)): lambda op, x, y: torch.add(x, y),
    ("MatMul", (torch.Tensor, torch.Tensor)): lambda op, x, y: torch.matmul(x, y),
}
