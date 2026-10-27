"""Task data for the fix experiment: a repair set shown to the fixer, a held-out set it never sees.

Formula runs draw from the task's training distribution and label with the true target; Iris runs
split the 150 flowers 100/50, stratified. Domains of at most SMALL_DOMAIN points (xor) have no
separate held-out set: repair and held-out are both the whole domain.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.grade import out_of_range_inputs
from experiments.tasks import get_task, uniform_sample

from ..spec import Spec

REPAIR_N, HELDOUT_N, OOR_N = 2000, 20_000, 20_000
REPAIR_SEED, HELDOUT_SEED, OOR_SEED, ANCHOR_SEED = 100, 101, 102, 103
SMALL_DOMAIN = 16
N_FLOWERS_HELDOUT = 50


@dataclass
class RepairData:
    X_repair: np.ndarray
    y_repair: np.ndarray
    X_heldout: np.ndarray
    y_heldout: np.ndarray
    X_oor: np.ndarray | None
    y_oor: np.ndarray | None
    flowers: bool        # the held-out set is the held-out flowers of a dataset task
    whole_domain: bool   # repair = held-out = the whole (tiny) domain


def training_points(task, rng: np.random.Generator, n: int) -> np.ndarray:
    return task.sampler(rng, n) if task.sampler is not None else uniform_sample(task.spec, rng, n)


def repair_data(rd: Path) -> RepairData:
    from experiments.datasets import DATASET_TASKS, table

    spec = Spec.load(Path(rd) / "spec.json")
    task = get_task(spec.task)
    if spec.task in DATASET_TASKS:
        from sklearn.model_selection import train_test_split

        X, y = table(spec.task)
        Xr, Xh, yr, yh = train_test_split(X, y.astype(np.float64), test_size=N_FLOWERS_HELDOUT, stratify=y,
                                          random_state=0)
        return RepairData(Xr, yr, Xh, yh, None, None, flowers=True, whole_domain=False)
    if spec.is_discrete() and spec.domain_size() <= SMALL_DOMAIN:
        X = np.array(list(itertools.product(*[range(int(i.lo), int(i.hi) + 1) for i in spec.inputs])),
                     dtype=np.float64)
        y = task.target(X).astype(np.float64)
        return RepairData(X, y, X, y, None, None, flowers=False, whole_domain=True)
    Xr = training_points(task, np.random.default_rng(REPAIR_SEED), REPAIR_N)
    Xh = training_points(task, np.random.default_rng(HELDOUT_SEED), HELDOUT_N)
    Xo = out_of_range_inputs(task, np.random.default_rng(OOR_SEED))
    Xo = None if Xo is None else Xo[:OOR_N]
    yo = None if Xo is None else task.target(Xo).astype(np.float64)
    return RepairData(Xr, task.target(Xr).astype(np.float64), Xh, task.target(Xh).astype(np.float64), Xo, yo,
                      flowers=False, whole_domain=False)
