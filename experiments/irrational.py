"""Irrational-function tasks (design 2026-10-30-irrational-functions): targets that are not piecewise linear.

Kept out of TASKS, like the dataset tasks; get_task resolves them. FAMILY names each task's formula family for the
report only: the AST tool never imports this module.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from nndc.spec import OutputKind, Spec

from .tasks import Task, _inputs


def _r2(X: np.ndarray) -> np.ndarray:
    return X[:, 0] ** 2 + X[:, 1] ** 2


def _task(name: str, output: OutputKind, inputs: tuple, target, expected: str, seeds: tuple[int, ...]) -> Task:
    return Task(name, Spec(name, output, _inputs(*inputs)), target, expected,
                hidden=[8] if len(inputs) == 1 else [16], handcrafted=None, steps=6000, seeds=seeds)


X3 = (("x", "float", -3, 3), ("y", "float", -3, 3))
X2 = (("x", "float", -2, 2), ("y", "float", -2, 2))
DEV = (0, 1, 2)
HELD = (0, 1)

_DEV = [
    _task("exp", "regression", (("x", "float", -2, 2),), lambda X: np.exp(X[:, 0]), "exp(x0)", DEV),
    _task("log", "regression", (("x", "float", 0.1, 5),), lambda X: np.log(X[:, 0]), "log(x0)", DEV),
    _task("sqrt", "regression", (("x", "float", 0, 4),), lambda X: np.sqrt(X[:, 0]), "sqrt(x0)", DEV),
    _task("square", "regression", (("x", "float", -2, 2),), lambda X: X[:, 0] ** 2, "x0*x0", DEV),
    _task("recip", "regression", (("x", "float", 0.5, 4),), lambda X: 1 / X[:, 0], "1/x0", DEV),
    _task("circle", "binary", X3, lambda X: (_r2(X) < 4).astype(int), "x0*x0 + x1*x1 < 4", DEV),
    _task("dist", "regression", X3, lambda X: np.sqrt(_r2(X)), "sqrt(x0*x0 + x1*x1)", DEV),
    _task("bump", "regression", X3, lambda X: np.exp(-_r2(X)), "exp(-(x0*x0 + x1*x1))", DEV),
    _task("product", "regression", X2, lambda X: X[:, 0] * X[:, 1], "x0*x1", DEV),
]

# held-out tasks: written before any tool code; tool development must not use them
_HELD = [
    _task("decay", "regression", (("x", "float", -1, 3),), lambda X: 3 * np.exp(-0.5 * X[:, 0]),
          "3*exp(-0.5*x0)", HELD),
    _task("logshift", "regression", (("x", "float", -1.5, 4),), lambda X: np.log(X[:, 0] + 2), "log(x0 + 2)", HELD),
    _task("ellipse", "binary", X3, lambda X: ((X[:, 0] - 0.5) ** 2 / 4 + X[:, 1] ** 2 < 1).astype(int),
          "(x0 - 0.5)*(x0 - 0.5)/4 + x1*x1 < 1", HELD),
    _task("hypot1", "regression", X2, lambda X: np.sqrt(_r2(X) + 1), "sqrt(x0*x0 + x1*x1 + 1)", HELD),
]

IRRATIONAL_TASKS: dict[str, Task] = {t.name: t for t in _DEV}
IRRATIONAL_HELDOUT_TASKS: dict[str, Task] = {t.name: t for t in _HELD}

FAMILY = {"exp": "exp", "log": "log", "sqrt": "sqrt", "square": "square", "recip": "reciprocal", "circle": "conic",
          "dist": "sqrt", "bump": "exp", "product": "quadratic",
          "decay": "exp", "logshift": "log", "ellipse": "conic", "hypot1": "sqrt"}


def is_irrational(name: str) -> bool:
    return name in IRRATIONAL_TASKS or name in IRRATIONAL_HELDOUT_TASKS


def widened(task: Task) -> Task:
    """The wide ReLU variant: 4x every hidden size, 3x the training steps."""
    assert task.hidden is not None
    return replace(task, hidden=[4 * h for h in task.hidden], steps=3 * task.steps)
