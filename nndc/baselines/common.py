"""Shared pieces of the step-4 baselines: samples of the net, checker-guided selection, info files."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments.tasks import uniform_sample

from ..check import check_program, net_output
from ..lang import ProgramError, parse_program, program_size
from ..model import MLP
from ..spec import Spec

N_SAMPLES = 50_000
BASELINE_VERSIONS = ["tree", "regions", "pysr", "pysr_snap"]


def net_samples(spec: Spec, mlp: MLP, n: int = N_SAMPLES, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs over the spec box (the full grid when the discrete domain fits in n), the net's raw output
    (value, score out - threshold, or argmax logits) and its decision."""
    if spec.is_discrete() and spec.domain_size() <= n:
        axes = [range(int(i.lo), int(i.hi) + 1) for i in spec.inputs]
        X = np.array(list(itertools.product(*axes)), dtype=np.float64)
    else:
        X = uniform_sample(spec, np.random.default_rng(seed), n)
    dec, score = net_output(mlp, spec, X)
    raw = mlp.forward(X.astype(np.float32)).astype(np.float64) if spec.output == "argmax" else score
    return X, raw, dec


def smallest_passing(cands: list[str], mlp: MLP, spec: Spec) -> tuple[int | None, list[dict]]:
    """The passing candidate with the fewest nodes (exactness breaks ties), as in the AST tool."""
    rows = []
    for src in cands:
        try:
            nodes = program_size(parse_program(src, spec.arg_names))["nodes"]
        except ProgramError as e:
            rows.append({"snap_pass": False, "exact_pass": False, "nodes": None, "error": str(e)})
            continue
        r = check_program(src, mlp, spec)
        rows.append({"snap_pass": r.snap_pass, "exact_pass": r.exact_pass, "nodes": nodes, "error": r.error})
    passing = [i for i, r in enumerate(rows) if r["snap_pass"]]
    if not passing:
        return None, rows
    return min(passing, key=lambda i: (rows[i]["nodes"], not rows[i]["exact_pass"])), rows


def write_info(rd: Path, name: str, info: dict) -> None:
    (Path(rd) / f"{name}_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
