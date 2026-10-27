"""Hand-built nets with known weights, to test the tool (not learned behaviour)."""
from __future__ import annotations

import numpy as np

from nndc.model import MLP


def _mlp(*layers: tuple[list, list]) -> MLP:
    return MLP([np.array(W, np.float32) for W, _ in layers], [np.array(b, np.float32) for _, b in layers])


def xor() -> MLP:
    # h0 = relu(x0 + x1), h1 = relu(x0 + x1 - 1); out = h0 - 2*h1 - 0.5
    return _mlp(([[1, 1], [1, 1]], [0, -1]), ([[1, -2]], [-0.5]))


def max_() -> MLP:
    # relu(a-b) + relu(b) - relu(-b)
    return _mlp(([[1, -1], [0, 1], [0, -1]], [0, 0, 0]), ([[1, 1, -1]], [0]))


def absdiff() -> MLP:
    return _mlp(([[1, -1], [-1, 1]], [0, 0]), ([[1, 1]], [0]))


def absdiff_perturbed() -> MLP:
    rng = np.random.default_rng(0)
    m = absdiff()
    for W, b in zip(m.weights, m.biases):
        W *= (1 + rng.uniform(-0.03, 0.03, W.shape)).astype(np.float32)
        b += rng.uniform(-0.01, 0.01, b.shape).astype(np.float32)
    return m


def triangle() -> MLP:
    return _mlp(([[1], [1], [1]], [-2, -3, -4]), ([[1, -2, 1]], [0]))


def ramp() -> MLP:
    return _mlp(([[1], [1]], [0, -1]), ([[1, -1]], [0]))


def window() -> MLP:
    # 0.5 - |x|
    return _mlp(([[1], [-1]], [0, 0]), ([[-1, -1]], [0.5]))


def neareq() -> MLP:
    # 0.2 - |a-b|
    return _mlp(([[1, -1], [-1, 1]], [0, 0]), ([[-1, -1]], [0.2]))


def deadredundant() -> MLP:
    # relu(a+b) always on, relu(-a-b-1) always off, relu(a-b) twice
    return _mlp(([[1, 1], [-1, -1], [1, -1], [1, -1]], [0, -1, 0, 0]), ([[1, 5, 0.5, 0.5]], [0]))


def sparse8() -> MLP:
    w = [0.0005] * 8
    w[2], w[5] = 1.0, -1.0
    return _mlp(([w], [0]), ([[1]], [0]))


def sum8() -> MLP:
    return _mlp(([[0.02] * 8], [-20]), ([[1]], [0]))
