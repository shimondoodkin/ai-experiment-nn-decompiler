"""Kink merge for one-layer sigmoid nets.

Each sigmoid becomes PIECES straight pieces (chords on knots uniform in its pre-activation over the input box),
so the net becomes an affine part plus ReLU kinks. Kinks on the same line add up, and many nearly cancel; a few
strong lines, refined and refitted to the net, give a short program. It counts only if it passes the checker.

Design: docs/design/2026-10-29-kink-merge-design.md (parameters fixed there before any tool code).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..model import MLP, activate
from ..spec import Spec

PIECES = 17
COS_MIN = 0.995                 # normals at least this parallel lie on one line ...
MERGE_OFFS = (0.02, 0.1)        # ... when their offsets (unit box) differ by less than this
MAX_LINES = 8
OFF_RANGE = 0.1                 # offset search: +-OFF_RANGE in OFF_STEPS steps, OFF_ROUNDS rounds
OFF_STEPS = 21
OFF_ROUNDS = 3
FIT_SAMPLES = 20_000
FIT_SEED = 0
CANCEL_FRAC = 0.25              # a merged line nearly cancels when |strength| < CANCEL_FRAC * sum |strengths|


@dataclass
class Kink:
    normal: np.ndarray          # unit-box coordinates, |normal| = 1, first nonzero component positive
    offset: float
    strength: float             # term: strength * max(normal @ u + offset, 0)


@dataclass
class Expansion:
    const: float
    linear: np.ndarray
    kinks: list[Kink]


@dataclass
class Line:
    normal: np.ndarray
    offset: float
    strength: float
    gross: float                # sum of the merged kinks' |strength|
    count: int


def applies(spec: Spec, mlp: MLP) -> str | None:
    """None when the route applies to this net, else the reason it does not."""
    if spec.output == "argmax":
        return "argmax output"
    if len(mlp.weights) != 2:
        return "not one hidden layer"
    if mlp.acts != ["sigmoid"]:
        return "not all sigmoid"
    return None


def unit_box(spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([float(i.lo) for i in spec.inputs])
    return lo, np.array([float(i.hi) for i in spec.inputs]) - lo


def _corners(n: int) -> np.ndarray:
    return np.array(list(itertools.product([0.0, 1.0], repeat=n)))


def expand(spec: Spec, mlp: MLP, pieces: int = PIECES) -> Expansion:
    lo, span = unit_box(spec)
    W, b = mlp.weights[0].astype(np.float64), mlp.biases[0].astype(np.float64)
    v, c = mlp.weights[1][0].astype(np.float64), float(mlp.biases[1][0])
    Wu, bu = W * span, W @ lo + b                       # z = Wu @ u + bu
    corners = _corners(len(lo))
    const, linear, kinks = c, np.zeros(len(lo)), []
    for j in range(len(v)):
        zc = corners @ Wu[j] + bu[j]
        if zc.max() - zc.min() <= 1e-12:                # constant pre-activation: a constant unit
            const += v[j] * float(activate("sigmoid", np.array([bu[j]]))[0])
            continue
        t = np.linspace(zc.min(), zc.max(), pieces + 1)
        s = np.diff(activate("sigmoid", t)) / np.diff(t)
        # chord interpolation: sig(t0) + s0 (z - t0) + sum_i (s_i - s_{i-1}) max(z - t_i, 0)
        const += v[j] * (float(activate("sigmoid", t[:1])[0]) + s[0] * (bu[j] - t[0]))
        linear = linear + v[j] * s[0] * Wu[j]
        n = float(np.linalg.norm(Wu[j]))
        for i in range(1, pieces):
            normal, offset, st = Wu[j] / n, (bu[j] - t[i]) / n, v[j] * (s[i] - s[i - 1]) * n
            if normal[np.flatnonzero(np.abs(normal) > 1e-12)[0]] < 0:     # st max(y,0) = st max(-y,0) + st y
                const += st * offset
                linear = linear + st * normal
                normal, offset = -normal, -offset
            kinks.append(Kink(normal, float(offset), float(st)))
    return Expansion(float(const), linear, kinks)


def evaluate_expansion(e: Expansion, U: np.ndarray) -> np.ndarray:
    y = e.const + U @ e.linear
    for k in e.kinks:
        y = y + k.strength * np.maximum(U @ k.normal + k.offset, 0.0)
    return y


def merge_lines(kinks: list[Kink], off: float, cos_min: float = COS_MIN) -> list[Line]:
    """Kinks on one line (greedy first match) add up; normal and offset are |strength|-weighted means."""
    groups: list[tuple[np.ndarray, float, list[Kink]]] = []
    for k in kinks:
        for g in groups:
            if float(g[0] @ k.normal) > cos_min and abs(g[1] - k.offset) < off:
                g[2].append(k)
                break
        else:
            groups.append((k.normal, k.offset, [k]))
    lines = []
    for _, _, ks in groups:
        w = np.array([abs(k.strength) for k in ks]) + 1e-300
        normal = sum(wi * k.normal for wi, k in zip(w, ks))
        normal = normal / np.linalg.norm(normal)
        offset = float(sum(wi * k.offset for wi, k in zip(w, ks)) / w.sum())
        lines.append(Line(normal, offset, float(sum(k.strength for k in ks)),
                          float(sum(abs(k.strength) for k in ks)), len(ks)))
    return sorted(lines, key=lambda ln: -abs(ln.strength))


def nearly_cancelled(line: Line) -> bool:
    return line.count >= 2 and abs(line.strength) < CANCEL_FRAC * line.gross
