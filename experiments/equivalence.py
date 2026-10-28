"""Equivalence check for refactors: re-run the pipeline on a copy of runs/ and runs_bugs/ and compare to the repo.

    uv run python -m experiments.equivalence copy     --copy DIR   # copy runs/ and runs_bugs/ into DIR
    uv run python -m experiments.equivalence evaluate --copy DIR   # evaluate_run on every run in the copy
    uv run python -m experiments.equivalence ast      --copy DIR   # run_ast on the non-held-out runs in the copy
    uv run python -m experiments.equivalence repair   --copy DIR   # repair baselines + evaluation in the copy
    uv run python -m experiments.equivalence compare  --copy DIR [--skip-glob PATTERN ...]
    uv run python -m experiments.equivalence redump                # dump.py / weights.npz vs the loaded nets (repo)

Run from the repo root. `compare` and `redump` exit 1 when anything differs; the batch commands exit 1 when a
run errored (one bad run is printed and skipped, as in nndc/cli.py).
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("runs", "runs_bugs")
COMPARED = {"results.json", "ast.py", "ast_trace.json", "dump.py"}   # by name, anywhere under a root
IRIS = ("iris", "iris_petal")
BATCH_ERRORS = (ValueError, KeyError, OSError)   # one bad run never stops the batch


def _copy_runs(args) -> int:
    dst = Path(args.copy).resolve()
    if dst == REPO or REPO in dst.parents or dst in REPO.parents:
        sys.exit(f"equivalence copy: refusing to delete {dst}: it is, lies inside, or contains the repo ({REPO})")
    if dst.exists():
        shutil.rmtree(dst)
    for root in ROOTS:
        shutil.copytree(REPO / root, dst / root)
        print(f"copied {root}/ -> {(dst / root).as_posix()}", flush=True)
    return 0


def _copy_runs_of(args) -> list[Path]:
    from nndc.run import all_runs

    dst = Path(args.copy)
    return [rd for root in ROOTS for rd in all_runs(dst / root)]


def _verdict(res: dict, version: str) -> str:
    v = res["versions"][version]
    c = v["check"]
    return "ERROR" if c["error"] else "PASS exact" if c["exact_pass"] else "PASS snap" if c["snap_pass"] else "FAIL"


def _batch(runs: list[Path], fn) -> int:
    """Call fn(rd) -> str for each run; print its line or the error. Exit code 1 if any run errored."""
    errors = 0
    for rd in runs:
        try:
            line = fn(rd)
        except BATCH_ERRORS as e:
            errors += 1
            print(f"{rd.as_posix()}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        print(f"{rd.as_posix()}: {line}", flush=True)
    print(f"{len(runs)} runs, {errors} errors")
    return 1 if errors else 0


def _evaluate(args) -> int:
    from nndc.run import evaluate_run

    def one(rd: Path) -> str:
        res = evaluate_run(rd)
        return ", ".join(f"{k} {_verdict(res, k)}" for k in res["versions"])

    return _batch(_copy_runs_of(args), one)


def _ast(args) -> int:
    from nndc.ast_tool import run_ast
    from nndc.run import all_runs, is_step1

    runs = [rd for rd in all_runs(Path(args.copy) / "runs")
            if (is_step1(rd) and rd.name != "seed2") or (rd.parent.name in IRIS and rd.name in ("seed0", "seed1", "seed2"))]
    return _batch(runs, lambda rd: _verdict(run_ast(rd), "ast"))


def _repair(args) -> int:
    from nndc.repair.run import evaluate_repair, run_repair_baselines
    from nndc.run import all_runs

    dst = Path(args.copy)
    runs = [rd for root in ROOTS for rd in all_runs(dst / root)
            if (REPO / root / rd.relative_to(dst / root) / "repair").is_dir()]

    def one(rd: Path) -> str:
        run_repair_baselines(rd)   # trains the weight-space arms and the tree, then evaluates them
        res = evaluate_repair(rd)
        return f"{len(res)} arms"

    return _batch(runs, one)


def _same_npz(a: Path, b: Path) -> bool:
    with np.load(a) as za, np.load(b) as zb:
        return sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def _same_text(a: Path, b: Path) -> bool:
    return a.read_bytes().replace(b"\r", b"") == b.read_bytes().replace(b"\r", b"")


def _compared_files(base: Path) -> set[Path]:
    """Relative paths of the compared files under `base` (a folder holding runs/ and runs_bugs/)."""
    out = set()
    for root in ROOTS:
        for p in (base / root).rglob("*"):
            if p.is_file() and (p.name in COMPARED or (p.parent.name == "repair" and fnmatch.fnmatch(p.name, "*_net.npz"))):
                out.add(p.relative_to(base))
    return out


def _compare(args) -> int:
    dst = Path(args.copy)
    rels = sorted(_compared_files(dst) | _compared_files(REPO))
    diff, skipped = [], 0
    for rel in rels:
        posix = rel.as_posix()
        if any(fnmatch.fnmatch(posix, pat) for pat in args.skip_glob):
            skipped += 1
            continue
        a, b = dst / rel, REPO / rel
        if not a.exists() or not b.exists():
            diff.append(f"{posix} (missing in {'copy' if not a.exists() else 'repo'})")
        elif not (_same_npz(a, b) if rel.suffix == ".npz" else _same_text(a, b)):
            diff.append(posix)
    for d in diff:
        print(f"DIFF {d}")
    print(f"compared {len(rels) - skipped} files ({skipped} skipped): {len(diff)} differ")
    return 1 if diff else 0


def _redump(args) -> int:
    from nndc.dump import dump_program
    from nndc.model import save_npz
    from nndc.run import all_runs, load_run

    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for root in ROOTS:
            for rd in all_runs(REPO / root):
                rel = rd.relative_to(REPO).as_posix()
                try:
                    spec, mlp = load_run(rd)
                    problems = []
                    if dump_program(mlp, spec).replace("\r", "") != (rd / "dump.py").read_text(encoding="utf-8").replace("\r", ""):
                        problems.append("dump.py")
                    tmp_npz = Path(tmp) / "weights.npz"
                    save_npz(mlp, tmp_npz)
                    if tmp_npz.read_bytes() != (rd / "weights.npz").read_bytes():
                        problems.append("weights.npz")
                except BATCH_ERRORS as e:
                    bad += 1
                    print(f"{rel}: ERROR {type(e).__name__}: {e}", flush=True)
                    continue
                if problems:
                    bad += 1
                    print(f"DIFF {rel}: {', '.join(problems)}", flush=True)
    print(f"redump: {bad} runs differ")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="equivalence", description="Prove a refactor changes no existing result.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in [
        ("copy", _copy_runs, "copy runs/ and runs_bugs/ into DIR (deleting DIR first)"),
        ("evaluate", _evaluate, "evaluate_run on every run in the copy"),
        ("ast", _ast, "run_ast on the non-held-out runs in the copy"),
        ("repair", _repair, "repair baselines + evaluation on the runs in the copy that have a repair/ folder"),
        ("compare", _compare, "compare the copy's result files with the repo's; exit 1 if any differ"),
    ]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--copy", required=True, metavar="DIR")
        if name == "compare":
            s.add_argument("--skip-glob", action="append", default=[], metavar="PATTERN",
                           help="leave out files whose relative path matches (repeatable)")
        s.set_defaults(func=fn)
    s = sub.add_parser("redump", help="re-dump every repo run's net and compare with dump.py and weights.npz")
    s.set_defaults(func=_redump)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
