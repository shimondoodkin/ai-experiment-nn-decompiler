"""Grade a program against the true task (report only; never used for acceptance)."""
from __future__ import annotations

import itertools

import numpy as np

from nndc.check import EXHAUSTIVE_CAP, program_decision
from nndc.lang import ProgramError, evaluate, parse_program
from nndc.spec import DISCRETE_TYPES

from .tasks import Task, uniform_sample

N_GRADE = 200_000


def _in_range(task: Task, rng: np.random.Generator) -> np.ndarray:
    spec = task.spec
    if spec.is_discrete() and spec.domain_size() <= EXHAUSTIVE_CAP:
        axes = [range(int(i.lo), int(i.hi) + 1) for i in spec.inputs]
        return np.array(list(itertools.product(*axes)), dtype=np.float64)
    return uniform_sample(spec, rng, N_GRADE)


def _out_of_range(task: Task, rng: np.random.Generator) -> np.ndarray | None:
    spec = task.spec
    cont = np.array([i.type not in DISCRETE_TYPES for i in spec.inputs])
    if not cont.any():
        return None
    lo = np.array([i.lo for i in spec.inputs], dtype=np.float64)
    hi = np.array([i.hi for i in spec.inputs], dtype=np.float64)
    pad = np.where(cont, 0.5 * (hi - lo), 0)
    X = uniform_sample(spec, rng, 4 * N_GRADE)
    X[:, cont] = rng.uniform(lo - pad, hi + pad, X.shape)[:, cont]
    outside = ((X < lo) | (X > hi)).any(axis=1)
    return X[outside][:N_GRADE]


def _score(task: Task, raw: np.ndarray, X: np.ndarray) -> tuple[float, bool]:
    y = task.target(X).astype(np.float64)
    if task.spec.output == "regression":
        err = np.where(np.isfinite(raw), np.abs(raw - y), np.inf)
        rng_y = float(y.max() - y.min()) or 1.0
        m = float(err.max())
        return m, m <= 1e-6 * rng_y
    acc = float((program_decision(task.spec, raw) == y).mean())
    return acc, acc == 1.0


def grade_task(src: str, task: Task, seed: int = 0) -> dict:
    metric = "max_abs_err" if task.spec.output == "regression" else "accuracy"
    rng = np.random.default_rng(seed)
    try:
        prog = parse_program(src, task.spec.arg_names)
        Xin = _in_range(task, rng)
        s_in, ok_in = _score(task, evaluate(prog, Xin), Xin)
        Xout = _out_of_range(task, rng)
        s_out, ok_out = (None, True) if Xout is None else _score(task, evaluate(prog, Xout), Xout)
    except ProgramError as e:
        return {"in_range": None, "out_of_range": None, "metric": metric, "recovered": False, "error": str(e)}
    return {"in_range": s_in, "out_of_range": s_out, "metric": metric, "recovered": bool(ok_in and ok_out),
            "error": None}
