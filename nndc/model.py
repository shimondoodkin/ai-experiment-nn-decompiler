"""Dense MLP (ReLU, sigmoid or tanh hidden layers) with a NumPy float32 reference forward pass.

Only stdlib + NumPy at module level: this module is copied into blind folders.
torch is imported inside from_torch/to_torch only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACTIVATIONS = ("relu", "sigmoid", "tanh")
# Lipschitz constant of each activation
LIPSCHITZ = {"relu": 1.0, "sigmoid": 0.25, "tanh": 1.0}


class UnsupportedModel(ValueError):
    pass


def activate(act: str, z: np.ndarray) -> np.ndarray:
    """Apply a hidden-layer activation; the result keeps z's dtype."""
    if act == "relu":
        return np.maximum(z, z.dtype.type(0))
    if act == "sigmoid":
        with np.errstate(under="ignore"):   # exp(-|z|) underflowing to 0 is exact enough
            e = np.exp(-np.abs(z))
            return np.where(z >= 0, 1 / (1 + e), e / (1 + e))
    if act == "tanh":
        return np.tanh(z)
    raise ValueError(f"unknown activation {act!r} (expected one of {ACTIVATIONS})")


@dataclass
class MLP:
    weights: list[np.ndarray]   # W_i, shape (out_i, in_i), float32
    biases: list[np.ndarray]    # b_i, shape (out_i,), float32
    acts: list[str] | None = None   # activation after every layer except the last; default all relu

    def __post_init__(self):
        n_hidden = len(self.weights) - 1
        if self.acts is None:
            self.acts = ["relu"] * n_hidden
        else:
            self.acts = [str(a) for a in self.acts]
            if len(self.acts) != n_hidden:
                raise ValueError(f"acts has {len(self.acts)} entries, expected {n_hidden}")
            for a in self.acts:
                if a not in ACTIVATIONS:
                    raise ValueError(f"unknown activation {a!r} (expected one of {ACTIVATIONS})")

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
            h = activate(self.acts[i], z) if i < len(self.weights) - 1 else z
        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.preactivations(X)[-1]

    def hidden(self, X: np.ndarray) -> np.ndarray:
        """Float64 output of the last hidden layer (X itself when there is no hidden layer)."""
        pre = self.preactivations(X)
        if len(pre) == 1:
            return np.asarray(X, dtype=np.float64)
        return activate(self.acts[-1], pre[-2]).astype(np.float64)   # activate in the net's float32, then cast


def from_torch(module) -> MLP:
    from torch import nn

    layers = list(module.children()) if isinstance(module, nn.Sequential) else [module]
    weights, biases, acts = [], [], []
    expect_linear = True
    for idx, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            if not expect_linear:
                raise UnsupportedModel(f"layer {idx}: two Linear layers in a row (expected an activation between)")
            weights.append(layer.weight.detach().cpu().numpy().astype(np.float32).copy())
            b = layer.bias
            biases.append(np.zeros(layer.out_features, np.float32) if b is None
                          else b.detach().cpu().numpy().astype(np.float32).copy())
            expect_linear = False
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            if expect_linear:
                raise UnsupportedModel(f"layer {idx}: {type(layer).__name__} must follow a Linear layer")
            acts.append(type(layer).__name__.lower())
            expect_linear = True
        else:
            raise UnsupportedModel(f"layer {idx}: unsupported layer type {type(layer).__name__}"
                                   " (only Linear, ReLU, Sigmoid and Tanh are supported)")
    if not weights or expect_linear:
        raise UnsupportedModel("model must end with a Linear layer")
    return MLP(weights, biases, acts)


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
            mods.append({"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[mlp.acts[i]]())
    return nn.Sequential(*mods)


def save_npz(mlp: MLP, path: Path) -> None:
    arrays = {}
    for i, (W, b) in enumerate(zip(mlp.weights, mlp.biases)):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
    if any(a != "relu" for a in mlp.acts):
        arrays["acts"] = np.array(mlp.acts)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def load_npz(path: Path) -> MLP:
    with np.load(path) as z:
        n = len([k for k in z.files if k.startswith("W")])
        acts = [str(a) for a in z["acts"]] if "acts" in z.files else None
        return MLP([z[f"W{i}"].astype(np.float32) for i in range(n)],
                   [z[f"b{i}"].astype(np.float32) for i in range(n)], acts)
