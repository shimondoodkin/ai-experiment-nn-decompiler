"""One folder per model: build the net, dump it, check and grade every program version."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from experiments.grade import grade_task
from experiments.tasks import get_task

from .check import CheckResult, check_program
from .dump import dump_program
from .lang import ProgramError, parse_program, program_size, read_program
from .model import MLP, load_npz, save_npz, to_torch
from .spec import Spec

RUNS = Path("runs")
AGENT_RE = re.compile(r"^agent_(\d+)\.py$")


def run_dir(task: str, seed: int | None, root: Path = RUNS) -> Path:
    return Path(root) / task / ("hand" if seed is None else f"seed{seed}")


def load_run(rd: Path) -> tuple[Spec, MLP]:
    return Spec.load(rd / "spec.json"), load_npz(rd / "weights.npz")


def all_runs(root: Path = RUNS) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/spec.json"))


def agent_attempts(rd: Path) -> list[tuple[int, Path]]:
    found = [(int(m.group(1)), p) for p in rd.glob("agent_*.py") if (m := AGENT_RE.match(p.name))]
    return sorted(found)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_run(task: str, seed: int | None, root: Path = RUNS, retrain: bool = False) -> Path:
    t = get_task(task)
    if seed is None and t.handcrafted is None:
        raise ValueError(f"task {task} has no handcrafted net")
    if seed is not None and t.hidden is None:
        raise ValueError(f"task {task} is hand-only (no training setup)")
    rd = run_dir(task, seed, root)
    attempts = agent_attempts(rd) if rd.exists() else []
    if retrain and attempts:
        raise RuntimeError(f"{rd} has agent attempts ({', '.join(p.name for _, p in attempts)}) that belong to "
                           "the current weights; move them away before retraining")
    if retrain or not (rd / "weights.npz").exists():
        rd.mkdir(parents=True, exist_ok=True)
        if seed is None:
            mlp, log = t.handcrafted(), {"kind": "hand"}
        else:
            import torch

            from experiments.train import train
            mlp, log = train(t, seed)
            torch.save(to_torch(mlp).state_dict(), rd / "net.pt")
        t.spec.save(rd / "spec.json")
        save_npz(mlp, rd / "weights.npz")
        _write_json(rd / "train_log.json", log)
    spec, mlp = load_run(rd)
    (rd / "dump.py").write_text(dump_program(mlp, spec), encoding="utf-8")
    res = evaluate_run(rd)
    if not res["versions"]["dump"]["check"]["exact_pass"]:
        raise RuntimeError(f"decompiler bug: {rd / 'dump.py'} does not match the net exactly")
    return rd


def _load_counterexamples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def evaluate_run(rd: Path) -> dict:
    spec, mlp = load_run(rd)
    task = get_task(spec.task)
    cex_path = rd / "counterexamples.json"
    cex = _load_counterexamples(cex_path)
    extra = np.array([[c[n] for n in spec.arg_names] for c in cex], dtype=np.float64) if cex else None
    seen = {tuple(c[n] for n in spec.arg_names) for c in cex}
    versions = {"dump": rd / "dump.py"} | {f"agent_{n}": p for n, p in agent_attempts(rd)}
    out = {}
    for name, path in versions.items():
        meta_path = rd / f"{name}_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        try:
            src = read_program(path)
        except ProgramError as e:
            metric = "max_abs_err" if spec.output == "regression" else "accuracy"
            out[name] = {"check": CheckResult(False, False, "empirical", 0, error=str(e)).to_json(),
                         "task": {"in_range": None, "out_of_range": None, "metric": metric, "recovered": False,
                                  "error": str(e)},
                         "size": None, "meta": meta}
            continue
        r = check_program(src, mlp, spec, extra_inputs=extra)
        try:
            size = program_size(parse_program(src))
        except ProgramError:
            size = None
        out[name] = {"check": r.to_json(), "task": grade_task(src, task), "size": size, "meta": meta}
        for c in r.counterexamples:
            key = tuple(c["inputs"][n] for n in spec.arg_names)
            if key not in seen:
                seen.add(key)
                cex.append(c["inputs"])
    if cex:
        _write_json(cex_path, cex)
    log_path = rd / "train_log.json"
    res = {"task": spec.task, "run": f"{spec.task}/{rd.name}", "kind": "hand" if rd.name == "hand" else "trained",
           "train": json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else None,
           "versions": out}
    _write_json(rd / "results.json", res)
    return res
