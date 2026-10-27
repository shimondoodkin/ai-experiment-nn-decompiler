"""Dense ReLU MLP with a NumPy float32 reference forward pass.

Only stdlib + NumPy at module level: this module is copied into blind folders.
torch is imported inside from_torch/to_torch only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class UnsupportedModel(ValueError):
    pass


@dataclass
class MLP:
    weights: list[np.ndarray]   # W_i, shape (out_i, in_i), float32
    biases: list[np.ndarray]    # b_i, shape (out_i,), float32
    # ReLU after every layer except the last

    @property
    def n_inputs(self) -> int:
        return self.weights[0].shape[1]

    @property
    def n_outputs(self) -> int:
        return self.weights[-1].shape[0]

    def preactivations(self, X: np.ndarray) -> list[np.ndarray]:
        h = np.asarray(X, dtype=np.float32)
        out = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            out.append(z)
            h = np.maximum(z, np.float32(0)) if i < len(self.weights) - 1 else z
        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.preactivations(X)[-1]


def from_torch(module) -> MLP:
    from torch import nn

    layers = list(module.children()) if isinstance(module, nn.Sequential) else [module]
    weights, biases = [], []
    expect_linear = True
    for idx, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            if not expect_linear:
                raise UnsupportedModel(f"layer {idx}: two Linear layers in a row (expected ReLU between)")
            weights.append(layer.weight.detach().cpu().numpy().astype(np.float32).copy())
            b = layer.bias
            biases.append(np.zeros(layer.out_features, np.float32) if b is None
                          else b.detach().cpu().numpy().astype(np.float32).copy())
            expect_linear = False
        elif isinstance(layer, nn.ReLU):
            if expect_linear:
                raise UnsupportedModel(f"layer {idx}: ReLU must follow a Linear layer")
            expect_linear = True
        else:
            raise UnsupportedModel(f"layer {idx}: unsupported layer type {type(layer).__name__}"
                                   " (only Linear and ReLU are supported)")
    if not weights or expect_linear:
        raise UnsupportedModel("model must end with a Linear layer")
    return MLP(weights, biases)


def to_torch(mlp: MLP):
    import torch
    from torch import nn

    mods = []
    for i, (W, b) in enumerate(zip(mlp.weights, mlp.biases)):
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W))
            lin.bias.copy_(torch.from_numpy(b))
        mods.append(lin)
        if i < len(mlp.weights) - 1:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def save_npz(mlp: MLP, path: Path) -> None:
    arrays = {}
    for i, (W, b) in enumerate(zip(mlp.weights, mlp.biases)):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def load_npz(path: Path) -> MLP:
    with np.load(path) as z:
        n = len([k for k in z.files if k.startswith("W")])
        return MLP([z[f"W{i}"].astype(np.float32) for i in range(n)],
                   [z[f"b{i}"].astype(np.float32) for i in range(n)])
