"""The fix experiment's arms on one run folder, and their evaluation (repair/results.json).

Weight-space arms (fine-tune, anchored fine-tune, last-layer LP, retrain from scratch) are nets in
repair/<arm>.npz. Program arms are repair/r_tree.py (a tree fitted on the repair set) and the agents'
fix_<n>.py / fixio_<n>.py; each program arm is also compiled back into a net (reinsertion) and that
net is checked against the program.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..check import check_program
from ..compile_back import CompileError, compile_program
from ..lang import ProgramError, evaluate, parse_program, program_size, read_program
from ..model import MLP, load_npz, save_npz
from ..run import agent_attempts, load_run, record_git
from ..spec import Spec
from .data import RepairData, repair_data
from .metrics import added_tests, decide, repair_metrics
from .weights import anchors, finetune, lp_repair

ARMS_W = ["w_ft", "w_ft_anchor", "w_lp", "w_retrain"]
SMALL_CV = 20   # repair sets smaller than this are not cross-validated (xor: the 4-point domain)


def faithful_program(rd: Path) -> Path:
    """The decompiled program the fix starts from: the best valid agent attempt, else the AST tool's."""
    from ..report import _best_attempt

    res_path = Path(rd) / "results.json"
    versions = json.loads(res_path.read_text(encoding="utf-8"))["versions"] if res_path.exists() else {}
    best = _best_attempt(versions, valid_only=True)
    if best is None and (Path(rd) / "agent_1.py").exists():
        best = "agent_1"
    path = Path(rd) / f"{best}.py" if best else Path(rd) / "ast.py"
    if not path.exists():
        raise FileNotFoundError(f"{rd}: no decompiled program (agent_n.py or ast.py) to fix")
    return path


def _tol(spec: Spec, data: RepairData) -> float:
    return spec.snap_tol_frac * float(data.y_heldout.max() - data.y_heldout.min()) if spec.output == "regression" \
        else 0.0


def _blind(rd: Path) -> tuple[Spec, MLP]:
    spec, mlp = load_run(rd)
    return Spec.from_json(spec.blind()), mlp


def _fit_tree(spec: Spec, X: np.ndarray, y: np.ndarray):
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    from ..baselines.tree import LEAF_SWEEP

    cls = DecisionTreeRegressor if spec.output == "regression" else DecisionTreeClassifier
    sweep = [k for k in LEAF_SWEEP if k <= max(2, len(X))]
    if len(X) < SMALL_CV:   # too few points to cross-validate: smallest tree with the best fit
        fits = [(cls(max_leaf_nodes=k, random_state=0).fit(X, y).score(X, y), -k) for k in sweep]
        best = -max(fits)[1]
    else:
        cv = KFold(5, shuffle=True, random_state=0) if spec.output == "regression" \
            else StratifiedKFold(5, shuffle=True, random_state=0)
        scores = [(cross_val_score(cls(max_leaf_nodes=k, random_state=0), X, y, cv=cv).mean(), -k) for k in sweep]
        best = -max(scores)[1]
    return cls(max_leaf_nodes=best, random_state=0).fit(X, y)


def _train_arms(rd: Path, spec: Spec, mlp: MLP, data: RepairData, X: np.ndarray, y: np.ndarray) -> dict:
    """The non-agent arms trained on (X, y): nets and the tree program (None where infeasible)."""
    from ..baselines.tree import tree_to_program

    tol = _tol(spec, data)
    Xa, ya = anchors(rd, mlp, spec, data, tol)
    return {"w_ft": finetune(mlp, spec, X, y),
            "w_ft_anchor": finetune(mlp, spec, np.concatenate([X, Xa]), np.concatenate([y, ya])),
            "w_lp": lp_repair(mlp, spec, X, y, tol),
            "r_tree": tree_to_program(_fit_tree(spec, X, y), spec)}


def _decisions(spec: Spec, arm, X: np.ndarray) -> np.ndarray:
    if isinstance(arm, MLP):
        return decide(spec, arm.forward(X.astype(np.float32)))
    return decide(spec, evaluate(parse_program(arm, spec.arg_names), X))


def _cv(rd: Path, spec: Spec, mlp: MLP, data: RepairData) -> dict:
    """Dataset runs: the non-agent arms over 5 repeats of a stratified 2-fold split of all flowers."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    X = np.concatenate([data.X_repair, data.X_heldout])
    y = np.concatenate([data.y_repair, data.y_heldout])
    out: dict[str, dict[str, list]] = {}
    for tr, te in RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0).split(X, y):
        arms = _train_arms(rd, spec, mlp, data, X[tr], y[tr])
        orig = _decisions(spec, mlp, X[te])
        for name, arm in arms.items():
            if arm is None:
                continue
            m = repair_metrics(spec, orig, _decisions(spec, arm, X[te]), y[te], 0.0, whole_domain=False)
            for key in ("generalization", "drawdown"):
                if m[key] is not None:
                    out.setdefault(name, {}).setdefault(key, []).append(m[key])
    return {name: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for name, d in out.items()}


def run_repair_baselines(rd: Path) -> dict:
    rd = Path(rd)
    spec, mlp = _blind(rd)
    data = repair_data(rd)
    out_dir = rd / "repair"
    out_dir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    arms = _train_arms(rd, spec, mlp, data, data.X_repair, data.y_repair)
    if (rd / "defect.json").exists():
        from experiments.tasks import get_task
        from experiments.train import train

        arms["w_retrain"], _ = train(get_task(Spec.load(rd / "spec.json").task), int(rd.name.removeprefix("seed")))
    info = {"seconds": round(time.perf_counter() - t0, 2), "infeasible": []}
    for name, arm in arms.items():
        if arm is None:
            info["infeasible"].append(name)
            (out_dir / f"{name}.npz").unlink(missing_ok=True)
        elif isinstance(arm, MLP):
            save_npz(arm, out_dir / f"{name}.npz")
        else:
            (out_dir / f"{name}.py").write_text(arm, encoding="utf-8")
    if data.flowers:
        info["cv"] = _cv(rd, spec, mlp, data)
    (out_dir / "baselines_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    record_git(rd, "repair_baselines")
    return evaluate_repair(rd)


N_BOX = 20_000


def _reinsertion_match(spec: Spec, src: str, net: MLP, data: RepairData, tol: float) -> dict:
    """Does the compiled net decide like the program? On the task data (repair + held-out), and the share of
    uniform box points where they differ: nonzero only inside the ramps that stand in for jumps."""
    from experiments.tasks import uniform_sample

    def agree(X: np.ndarray) -> np.ndarray:
        p, n = _decisions(spec, src, X), _decisions(spec, net, X)
        return np.abs(p - n) <= tol if spec.output == "regression" else p == n

    task = agree(np.concatenate([data.X_repair, data.X_heldout]))
    box = agree(uniform_sample(spec, np.random.default_rng(0), N_BOX))
    return {"task_match": float(task.mean()), "box_mismatch": float(1 - box.mean())}


def _program_arms(rd: Path) -> dict[str, Path]:
    arms = {}
    if (rd / "repair" / "r_tree.py").exists():
        arms["r_tree"] = rd / "repair" / "r_tree.py"
    for prefix in ("fix", "fixio"):
        arms |= {f"{prefix}_{n}": p for n, p in agent_attempts(rd, prefix)}
    return arms


def evaluate_repair(rd: Path) -> dict:
    rd = Path(rd)
    spec, mlp = _blind(rd)
    data = repair_data(rd)
    tol = _tol(spec, data)
    out_dir = rd / "repair"
    out_dir.mkdir(exist_ok=True)
    sets = {"repair": (data.X_repair, data.y_repair), "heldout": (data.X_heldout, data.y_heldout)}
    if data.X_oor is not None:
        sets["oor"] = (data.X_oor, data.y_oor)
    orig = {k: _decisions(spec, mlp, X) for k, (X, _) in sets.items()}
    info_path = out_dir / "baselines_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    try:
        faithful = read_program(faithful_program(rd))
    except FileNotFoundError:
        faithful = None

    def metrics(arm) -> dict:
        res = {}
        for k, (X, y) in sets.items():
            res[k] = repair_metrics(spec, orig[k], _decisions(spec, arm, X), y, tol, data.whole_domain)
        return res

    results: dict[str, dict] = {}
    for name in ARMS_W:
        path = out_dir / f"{name}.npz"
        if path.exists():
            results[name] = {**metrics(load_npz(path)), "nodes": None, "added_tests": None, "reinsertion": None,
                             "error": None}
        elif name in info.get("infeasible", []) or (name != "w_retrain" and info):
            results[name] = {"error": "infeasible" if name in info.get("infeasible", []) else "missing"}
    for name, path in _program_arms(rd).items():
        try:
            src = read_program(path)
            row = {**metrics(src), "nodes": program_size(parse_program(src, spec.arg_names))["nodes"],
                   "added_tests": added_tests(faithful, src, data.X_repair, spec.arg_names) if faithful else None,
                   "error": None}
        except ProgramError as e:
            results[name] = {"error": str(e)}
            continue
        try:
            net, cinfo = compile_program(src, spec)
            save_npz(net, out_dir / f"{name}_net.npz")
            r = check_program(src, net, spec)
            row["reinsertion"] = {"exact_pass": r.exact_pass, "snap_pass": r.snap_pass, "ramps": cinfo["ramps"],
                                  "hidden": cinfo["hidden"], **_reinsertion_match(spec, src, net, data, tol)}
        except CompileError as e:
            row["reinsertion"] = {"error": str(e)}
        results[name] = row
    if "cv" in info:
        for name, cv in info["cv"].items():
            results.setdefault(name, {})["cv"] = cv
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
