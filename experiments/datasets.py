"""Dataset tasks: a fixed labelled table instead of a known formula (kept out of TASKS and the step-1 report).

The target on arbitrary inputs is the label of the nearest table row (1-NN), so grading has
something to compare against off the table; on the table it is the table's own label.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from nndc.spec import InputSpec, Spec

from .tasks import Task

DATA = Path(__file__).parent / "data"
IRIS_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@cache
def _iris() -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(DATA / "iris.csv", delimiter=",", skiprows=1)
    return raw[:, :4], raw[:, 4].astype(int)


def _iris_task(name: str, cols: list[int], hidden: list[int]) -> Task:
    X, y = _iris()
    X = X[:, cols]
    inputs = [InputSpec(f"x{k}", IRIS_COLS[c], "float", float(X[:, k].min()), float(X[:, k].max()))
              for k, c in enumerate(cols)]

    def target(Q: np.ndarray) -> np.ndarray:
        d = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        return y[np.argmin(d, axis=1)]

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return X[rng.integers(0, len(X), n)]

    return Task(name, Spec(name, "argmax", inputs, n_outputs=3), target, expected="", hidden=hidden,
                handcrafted=None, steps=3000, lr=1e-2, sampler=sampler, seeds=(0, 1, 2, 3, 4, 5))


DATASET_TASKS: dict[str, Task] = {t.name: t for t in [
    _iris_task("iris", [0, 1, 2, 3], [8]),
    _iris_task("iris_petal", [2, 3], [8]),
]}


def table(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The labelled rows behind a dataset task, in its input columns."""
    X, y = _iris()
    return (X, y) if name == "iris" else (X[:, [2, 3]], y)
