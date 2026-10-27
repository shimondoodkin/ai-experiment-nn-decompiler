"""Planted bugs for the fix experiment: train a formula task on a target that is wrong in one region.

The net learns the bug, so its decompiled program copies it faithfully, and the true formula is known
everywhere: a controlled "net is wrong" case. Regression targets get +20% of the output range inside
an axis-aligned sub-box of 10% of the input box; binary targets get their label flipped there. On an
all-discrete domain (xor) one domain point is flipped instead.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .tasks import Task, uniform_sample

BOX_FRACTION = 0.1
OFFSET_FRACTION = 0.2


@dataclass
class Defect:
    kind: str              # "offset" (regression) | "flip" (binary)
    box_lo: list[float]
    box_hi: list[float]    # equal to box_lo for a single discrete point
    offset: float = 0.0

    def inside(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return ((X >= np.array(self.box_lo)) & (X <= np.array(self.box_hi))).all(axis=1)

    def to_json(self) -> dict:
        return asdict(self)


def make_defect(task: Task, seed: int) -> Defect:
    spec = task.spec
    rng = np.random.default_rng(seed)
    kind = "offset" if spec.output == "regression" else "flip"
    offset = 0.0
    if kind == "offset":
        y = task.target(uniform_sample(spec, np.random.default_rng(0), 20_000))
        offset = OFFSET_FRACTION * float(y.max() - y.min())
    if spec.is_discrete():
        domain = list(itertools.product(*[range(int(i.lo), int(i.hi) + 1) for i in spec.inputs]))
        p = [float(v) for v in domain[int(rng.integers(len(domain)))]]
        return Defect(kind, p, list(p), offset)
    lo = np.array([i.lo for i in spec.inputs], dtype=np.float64)
    hi = np.array([i.hi for i in spec.inputs], dtype=np.float64)
    side = BOX_FRACTION ** (1 / lo.size) * (hi - lo)
    corner = rng.uniform(lo, hi - side)
    return Defect(kind, corner.tolist(), (corner + side).tolist(), offset)


def defective_target(task: Task, d: Defect) -> Callable[[np.ndarray], np.ndarray]:
    def target(X: np.ndarray) -> np.ndarray:
        y = task.target(X).astype(np.float64)
        m = d.inside(X)
        y[m] = 1 - y[m] if d.kind == "flip" else y[m] + d.offset
        return y
    return target
