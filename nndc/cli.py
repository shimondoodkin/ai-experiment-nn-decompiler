"""nndc command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import check_program, format_result
from .run import RUNS, build_run, load_run


def _train_summary(res: dict) -> str:
    t = res.get("train") or {}
    if t.get("kind") == "hand":
        return "handcrafted"
    q = t.get("quality", {})
    shown = ", ".join(f"{k}={v:.4g}" for k, v in q.items())
    return f"{shown}, learned={t.get('learned')}"


def cmd_run(args) -> int:
    from experiments.tasks import TASKS, get_task

    names = list(TASKS) if args.task == "all" else [args.task]
    for name in names:
        t = get_task(name)
        if args.hand:
            seeds = [None]
        elif args.seed is not None:
            seeds = [args.seed]
        else:
            seeds = ([None] if t.handcrafted else []) + ([0, 1, 2] if t.hidden else [])
        for seed in seeds:
            rd = build_run(name, seed, Path(args.root), retrain=args.retrain)
            res = json.loads((rd / "results.json").read_text(encoding="utf-8"))
            dump = res["versions"]["dump"]["check"]
            print(f"{rd.as_posix()}: {_train_summary(res)}; dump "
                  f"{'PASS exact' if dump['exact_pass'] else 'FAIL'} [{dump['label']}]", flush=True)
    return 0


def cmd_check(args) -> int:
    spec, mlp = load_run(Path(args.run))
    r = check_program(Path(args.program).read_text(encoding="utf-8"), mlp, spec)
    print(format_result(r, spec), end="")
    return 0 if r.snap_pass else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nndc", description="Decompile small ReLU nets into programs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build/train a model, dump it, check and grade every version")
    r.add_argument("task", help="task name or 'all'")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--seed", type=int)
    g.add_argument("--hand", action="store_true")
    r.add_argument("--retrain", action="store_true")
    r.add_argument("--root", default=str(RUNS))
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", help="check a program against a run's net")
    c.add_argument("program")
    c.add_argument("--run", required=True)
    c.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
