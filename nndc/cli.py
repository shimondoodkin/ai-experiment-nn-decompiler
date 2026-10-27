"""nndc command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import CheckResult, check_program, format_result
from .lang import ProgramError, read_program
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
    try:
        r = check_program(read_program(Path(args.program)), mlp, spec)
    except ProgramError as e:
        r = CheckResult(False, False, "empirical", 0, error=str(e))
    print(format_result(r, spec), end="")
    return 0 if r.snap_pass else 1


def cmd_agent_prepare(args) -> int:
    from .agent.prepare import prepare, suggested_commands
    from .run import all_runs

    runs = all_runs(Path(args.root)) if args.all else [Path(r) for r in args.runs]
    if not runs:
        print("nothing to prepare: give run folders or --all")
        return 2
    for rd in runs:
        b = prepare(rd)
        print(f"{rd.as_posix()} -> {b}")
        for c in suggested_commands(b):
            print(f"    {c}")
    return 0


def cmd_agent_collect(args) -> int:
    from .agent.collect import collect

    trs = [Path(t) for t in args.transcript] if args.transcript else None
    rd, n = collect(Path(args.blind), transcripts=trs)
    if n is None:
        print(f"{rd.as_posix()}: identical attempt already collected, nothing changed")
        return 0
    res = json.loads((rd / "results.json").read_text(encoding="utf-8"))
    v = res["versions"][f"agent_{n}"]
    c = v["check"]
    verdict = "ERROR" if c["error"] else "PASS exact" if c["exact_pass"] else "PASS snap" if c["snap_pass"] else "FAIL"
    meta = v.get("meta") or {}
    print(f"{rd.as_posix()}: agent_{n} {verdict}; recovered={v['task']['recovered']}; "
          f"transcript_found={meta.get('transcript_found')}; outside_paths={len(meta.get('outside_paths') or [])}")
    return 0


def cmd_agent_mock(args) -> int:
    from .agent.mock import mock_agent

    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else ""
    mock_agent(Path(args.blind), Path(args.program).read_text(encoding="utf-8"), notes)
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    Path(args.out).write_text(build_report(Path(args.root)), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def cmd_gate(args) -> int:
    from .report import spike_gate

    passed, lines = spike_gate(Path(args.root))
    print("\n".join(lines))
    return 0 if passed else 1


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

    a = sub.add_parser("agent", help="blind folders for Claude Code refactoring runs")
    asub = a.add_subparsers(dest="agent_cmd", required=True)
    ap = asub.add_parser("prepare", help="create blind folder(s) to run Claude Code in")
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default=str(RUNS))
    ap.set_defaults(func=cmd_agent_prepare)
    ac = asub.add_parser("collect", help="bring an attempt back, audit it and re-check it")
    ac.add_argument("blind")
    ac.add_argument("--transcript", action="append",
                    help="transcript JSONL to audit (e.g. a subagent's); default: look it up by folder")
    ac.set_defaults(func=cmd_agent_collect)
    am = asub.add_parser("mock", help="write a given program as a fake attempt (testing)")
    am.add_argument("blind")
    am.add_argument("--program", required=True)
    am.add_argument("--notes")
    am.set_defaults(func=cmd_agent_mock)

    rp = sub.add_parser("report", help="write REPORT.md from all runs")
    rp.add_argument("--root", default=str(RUNS))
    rp.add_argument("--out", default="REPORT.md")
    rp.set_defaults(func=cmd_report)

    gt = sub.add_parser("gate", help="evaluate the spike gate")
    gt.add_argument("--root", default=str(RUNS))
    gt.set_defaults(func=cmd_gate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
