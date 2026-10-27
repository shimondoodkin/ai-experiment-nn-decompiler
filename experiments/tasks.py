"""Step-1 tasks: specs, true targets and expected formulas (written before any run).

Targets are plain NumPy, written independently of `expected`, so tests can cross-check them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nndc.model import MLP
from nndc.spec import DISCRETE_TYPES, InputSpec, Spec

from . import handcrafted as hc


@dataclass
class Task:
    name: str
    spec: Spec                                  # real names filled in
    target: Callable[[np.ndarray], np.ndarray]  # (N,d) -> (N,): value, or class 0/1
    expected: str                               # program-language expression in x0.., written before any run
    hidden: list[int] | None                    # trained architecture; None = hand-only
    handcrafted: Callable[[], MLP] | None
    steps: int = 3000
    lr: float = 1e-2
    sampler: Callable[[np.random.Generator, int], np.ndarray] | None = None
    seeds: tuple[int, ...] = (0, 1, 2)          # trained seeds that `nndc run all` builds


def _inputs(*items: tuple[str, str, float, float]) -> list[InputSpec]:
    return [InputSpec(f"x{k}", real, typ, lo, hi) for k, (real, typ, lo, hi) in enumerate(items)]


def uniform_sample(spec: Spec, rng: np.random.Generator, n: int) -> np.ndarray:
    cols = []
    for i in spec.inputs:
        if i.type in DISCRETE_TYPES:
            cols.append(rng.integers(int(i.lo), int(i.hi) + 1, n).astype(np.float64))
        else:
            cols.append(rng.uniform(i.lo, i.hi, n))
    return np.stack(cols, axis=1)


def _xor_points(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)


def _neareq_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.uniform(-2, 2, n)
    near = np.clip(a + rng.uniform(-0.4, 0.4, n), -2, 2)
    b = np.where(np.arange(n) < n // 2, near, rng.uniform(-2, 2, n))
    return np.stack([a, b], axis=1)


def _tax(x: np.ndarray) -> np.ndarray:
    return 0.1 * np.minimum(x, 20) + 0.2 * np.clip(x - 20, 0, 60) + 0.4 * np.maximum(x - 80, 0)


AB3 = (("a", "float", -3, 3), ("b", "float", -3, 3))
AB2 = (("a", "float", -2, 2), ("b", "float", -2, 2))

_ALL = [
    Task("xor", Spec("xor", "binary", _inputs(("p", "bool", 0, 1), ("q", "bool", 0, 1))),
         lambda X: (X[:, 0] != X[:, 1]).astype(int), "x0 != x1",
         hidden=[4], handcrafted=hc.xor, lr=5e-2, sampler=_xor_points),
    Task("max", Spec("max", "regression", _inputs(*AB3)),
         lambda X: np.maximum(X[:, 0], X[:, 1]), "max(x0, x1)", hidden=None, handcrafted=hc.max_),
    Task("absdiff", Spec("absdiff", "regression", _inputs(*AB3)),
         lambda X: np.abs(X[:, 0] - X[:, 1]), "abs(x0 - x1)", hidden=[8], handcrafted=hc.absdiff),
    Task("absdiff_perturbed", Spec("absdiff_perturbed", "regression", _inputs(*AB3)),
         lambda X: np.abs(X[:, 0] - X[:, 1]), "abs(x0 - x1)", hidden=None, handcrafted=hc.absdiff_perturbed),
    Task("triangle", Spec("triangle", "regression", _inputs(("x", "float", -2, 6))),
         lambda X: np.maximum(0, 1 - np.abs(X[:, 0] - 3)), "max(0, 1 - abs(x0 - 3))",
         hidden=None, handcrafted=hc.triangle),
    Task("ramp", Spec("ramp", "regression", _inputs(("x", "float", -2, 3))),
         lambda X: np.clip(X[:, 0], 0, 1), "min(max(x0, 0), 1)", hidden=None, handcrafted=hc.ramp),
    Task("window", Spec("window", "binary", _inputs(("x", "float", -2, 2))),
         lambda X: (np.abs(X[:, 0]) < 0.5).astype(int), "abs(x0) < 0.5", hidden=[4], handcrafted=hc.window),
    Task("neareq", Spec("neareq", "binary", _inputs(*AB2)),
         lambda X: (np.abs(X[:, 0] - X[:, 1]) < 0.2).astype(int), "abs(x0 - x1) < 0.2",
         hidden=[8], handcrafted=hc.neareq, sampler=_neareq_sample),
    Task("deadredundant", Spec("deadredundant", "regression", _inputs(("a", "float", 0, 1), ("b", "float", 0, 1))),
         lambda X: X[:, 0] + X[:, 1] + np.maximum(0, X[:, 0] - X[:, 1]), "x0 + x1 + max(0, x0 - x1)",
         hidden=None, handcrafted=hc.deadredundant),
    Task("sparse8", Spec("sparse8", "regression", _inputs(*[(f"u{k}", "float", 0, 1) for k in range(8)])),
         lambda X: np.maximum(0, X[:, 2] - X[:, 5]), "max(0, x2 - x5)", hidden=None, handcrafted=hc.sparse8),
    Task("sum8", Spec("sum8", "regression", _inputs(*[(f"v{k}", "int", 0, 255) for k in range(8)])),
         lambda X: np.maximum(0, 0.02 * X.sum(axis=1) - 20),
         "max(0, 0.02 * sum(x0, x1, x2, x3, x4, x5, x6, x7) - 20)", hidden=None, handcrafted=hc.sum8),
    Task("median3", Spec("median3", "regression", _inputs(*AB3, ("c", "float", -3, 3))),
         lambda X: np.median(X, axis=1), "max(min(x0, x1), min(max(x0, x1), x2))",
         hidden=[16, 16], handcrafted=None),
    Task("tax", Spec("tax", "regression", _inputs(("income", "float", 0, 200))),
         lambda X: _tax(X[:, 0]), "0.1 * min(x0, 20) + 0.2 * min(max(x0 - 20, 0), 60) + 0.4 * max(x0 - 80, 0)",
         hidden=[8], handcrafted=None, steps=6000),
]

TASKS: dict[str, Task] = {t.name: t for t in _ALL}


def get_task(name: str) -> Task:
    from .datasets import DATASET_TASKS

    try:
        return TASKS[name] if name in TASKS else DATASET_TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known: {', '.join([*TASKS, *DATASET_TASKS])}") from None


def expected_program(task: Task) -> str:
    return f"def net({', '.join(task.spec.arg_names)}):\n    return {task.expected}\n"
