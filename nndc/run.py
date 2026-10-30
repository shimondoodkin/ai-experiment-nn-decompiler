"""One folder per model: build the net, dump it, check and grade every program version."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

from experiments.grade import grade_task
from experiments.tasks import TASKS, expected_program, get_task

from .baselines.common import BASELINE_VERSIONS
from .check import CheckResult, check_program
from .dump import dump_program
from .lang import ProgramError, parse_program, program_size, read_program
from .model import MLP, load_npz, save_npz, to_torch
from .spec import Spec

RUNS = Path("runs")
AGENT_RE = re.compile(r"^agent_(\d+)\.py$")
GIT_FILE = "run_git.txt"
GIT_HEADER = ("# code version each step ran with: <step>: <git commit>[+dirty] <date> [note]; "
              "+dirty = uncommitted changes in nndc/ or experiments/\n")


def git_version() -> str:
    """Short commit of the code, with +dirty when nndc/ or experiments/ have uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "nndc", "experiments"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def record_git(rd: Path, step: str, version: str | None = None, when: str | None = None, note: str = "") -> None:
    """Write `step: <commit> <date>` into the run's run_git.txt, replacing an earlier line for the same step."""
    path = Path(rd) / GIT_FILE
    old = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    keep = [ln for ln in old if ln.strip() and not ln.startswith("#") and not ln.startswith(f"{step}:")]
    line = f"{step}: {version or git_version()} {when or date.today().isoformat()}" + (f" {note}" if note else "")
    path.write_text(GIT_HEADER + "\n".join(keep + [line]) + "\n", encoding="utf-8")


def read_git(rd: Path) -> dict[str, str]:
    """{step: commit} from a run's run_git.txt (empty when there is none)."""
    path = Path(rd) / GIT_FILE
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ":" in ln and not ln.startswith("#"):
            step, rest = ln.split(":", 1)
            out[step.strip()] = rest.split()[0] if rest.split() else ""
    return out


def is_step1(rd: Path) -> bool:
    """A run of a step-1 task (formula known); dataset runs such as iris are not."""
    return Path(rd).parent.name in TASKS


def run_dir(task: str, seed: int | None, root: Path = RUNS) -> Path:
    return Path(root) / task / ("hand" if seed is None else f"seed{seed}")


def load_run(rd: Path) -> tuple[Spec, MLP]:
    return Spec.load(rd / "spec.json"), load_npz(rd / "weights.npz")


def all_runs(root: Path = RUNS) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/spec.json"))


def agent_attempts(rd: Path, prefix: str = "agent") -> list[tuple[int, Path]]:
    """Numbered attempts `<prefix>_<n>.py` (prefix agent, io_agent, fix or fixio)."""
    pat = AGENT_RE if prefix == "agent" else re.compile(rf"^{prefix}_(\d+)\.py$")
    found = [(int(m.group(1)), p) for p in rd.glob(f"{prefix}_*.py") if (m := pat.match(p.name))]
    return sorted(found)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _bug_learned(mlp: MLP, t, defect, target, seed: int) -> float:
    """Share of training-distribution points inside the defect where the net follows the bug, not the truth."""
    from experiments.train import _sample

    X = _sample(t, np.random.default_rng(2000 + seed), 20_000)
    X = X[defect.inside(X)]
    if len(X) == 0:
        return float("nan")
    out = mlp.forward(X.astype(np.float32)).astype(np.float64)[:, 0]
    bad, true = target(X), t.target(X).astype(np.float64)
    if t.spec.output == "binary":
        return float(((out > t.spec.threshold).astype(float) == bad).mean())
    return float((np.abs(out - bad) < np.abs(out - true)).mean())


def build_run(task: str, seed: int | None, root: Path = RUNS, retrain: bool = False, bug: bool = False,
              act: str = "relu") -> Path:
    """bug=True trains on a planted-bug target (experiments/defects.py, defect seed = seed); the spec keeps
    the true task, so grading measures the truth. act sets the hidden activation of a trained net; a non-ReLU
    run of a formula task also gets formula.py, the intended formula as a reference version."""
    t = get_task(task)
    if act != "relu" and (seed is None or bug):
        raise ValueError(f"act={act} needs a trained seed without a planted bug (hand nets are ReLU)")
    if bug and seed is None:
        raise ValueError("a planted bug needs a trained seed")
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
            if bug:
                from experiments.defects import defective_target, make_defect
                defect = make_defect(t, seed)
                target = defective_target(t, defect)
                mlp, log = train(t, seed, target)
                log["defect"] = defect.to_json()
                log["bug_learned"] = _bug_learned(mlp, t, defect, target, seed)
                _write_json(rd / "defect.json", defect.to_json())
            elif act == "relu":
                mlp, log = train(t, seed)
            else:
                mlp, log = train(t, seed, act=act)
            torch.save(to_torch(mlp).state_dict(), rd / "net.pt")
        t.spec.save(rd / "spec.json")
        save_npz(mlp, rd / "weights.npz")
        _write_json(rd / "train_log.json", log)
        record_git(rd, "build")
    spec, mlp = load_run(rd)
    if any(a != act for a in mlp.acts):
        raise ValueError(f"{rd} holds a net with activations {mlp.acts}, not act={act}; use another --root")
    (rd / "dump.py").write_text(dump_program(mlp, spec), encoding="utf-8")
    if any(a != "relu" for a in mlp.acts) and task in TASKS:
        (rd / "formula.py").write_text(expected_program(t), encoding="utf-8")
    res = evaluate_run(rd)
    if not res["versions"]["dump"]["check"]["exact_pass"]:
        raise RuntimeError(f"decompiler bug: {rd / 'dump.py'} does not match the net exactly")
    return rd


def _load_counterexamples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


OWN_CEX = ("ast4",)   # versions whose counterexamples go to counterexamples_<name>.json, not the shared file


def evaluate_run(rd: Path) -> dict:
    spec, mlp = load_run(rd)
    task = get_task(spec.task)
    cex_path = rd / "counterexamples.json"
    cex = _load_counterexamples(cex_path)
    extra = np.array([[c[n] for n in spec.arg_names] for c in cex], dtype=np.float64) if cex else None
    seen = {tuple(c[n] for n in spec.arg_names) for c in cex}
    versions = {"dump": rd / "dump.py"} | {f"agent_{n}": p for n, p in agent_attempts(rd)}
    if (rd / "ast.py").exists():
        versions["ast"] = rd / "ast.py"
    versions |= {name: rd / f"{name}.py" for name in OWN_CEX if (rd / f"{name}.py").exists()}
    versions |= {f"io_agent_{n}": p for n, p in agent_attempts(rd, "io_agent")}
    versions |= {name: rd / f"{name}.py" for name in BASELINE_VERSIONS if (rd / f"{name}.py").exists()}
    if (rd / "formula.py").exists():   # reference only: its counterexamples are not shared with other versions
        versions["formula"] = rd / "formula.py"
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
        own = None
        if name in OWN_CEX:   # a later tool version: checked on the shared set plus its own, adds only to its own
            own_path = rd / f"counterexamples_{name}.json"
            own = _load_counterexamples(own_path)
            more = np.array([[c[n] for n in spec.arg_names] for c in own], dtype=np.float64) if own else None
            ex = extra if more is None else more if extra is None else np.vstack([extra, more])
            r = check_program(src, mlp, spec, extra_inputs=ex)
        else:
            r = check_program(src, mlp, spec, extra_inputs=extra)
        try:
            size = program_size(parse_program(src))
        except ProgramError:
            size = None
        out[name] = {"check": r.to_json(), "task": grade_task(src, task), "size": size, "meta": meta}
        if own is not None:
            known = seen | {tuple(c[n] for n in spec.arg_names) for c in own}
            new = [c["inputs"] for c in r.counterexamples
                   if tuple(c["inputs"][n] for n in spec.arg_names) not in known]
            if new:
                _write_json(own_path, own + new)
            continue
        for c in ([] if name == "formula" else r.counterexamples):
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
