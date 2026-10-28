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
    from experiments.datasets import DATASET_TASKS
    from experiments.tasks import TASKS, get_task

    names = [*TASKS, *DATASET_TASKS] if args.task == "all" else [args.task]
    for name in names:
        t = get_task(name)
        if args.hand:
            seeds = [None]
        elif args.seed is not None:
            seeds = [args.seed]
        else:
            hand = [None] if t.handcrafted and args.act == "relu" else []   # hand nets are ReLU
            seeds = hand + (list(t.seeds) if t.hidden else [])
        for seed in seeds:
            rd = build_run(name, seed, Path(args.root), retrain=args.retrain, bug=args.bug, act=args.act)
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
    kind = "io" if args.io else "fix" if args.fix else "fixio" if args.fixio else "agent"
    for rd in runs:
        b = prepare(rd, kind=kind)
        print(f"{rd.as_posix()} -> {b}")
        for c in suggested_commands(b):
            print(f"    {c}")
    return 0


def cmd_agent_collect(args) -> int:
    from .agent import prepare as prep
    from .agent.collect import collect

    trs = [Path(t) for t in args.transcript] if args.transcript else None
    blind = Path(args.blind)
    kind = prep.load_index(prep.INDEX).get(blind.name, {}).get("kind", "agent")
    rd, n = collect(blind, index=prep.INDEX, transcripts=trs)
    if n is None:
        print(f"{rd.as_posix()}: identical attempt already collected, nothing changed")
        return 0
    name = f"{prep.KINDS[kind]}_{n}"
    meta = json.loads((rd / f"{name}_meta.json").read_text(encoding="utf-8"))
    audit = (f"transcript_found={meta.get('transcript_found')}; outside_paths={len(meta.get('outside_paths') or [])}"
             f"; forbidden_paths={len(meta.get('forbidden_paths') or [])}")
    if kind in ("fix", "fixio"):   # judged on task cases (repair/results.json), not against the net
        r = json.loads((rd / "repair" / "results.json").read_text(encoding="utf-8"))[name]
        h = r.get("heldout") or {}
        print(f"{rd.as_posix()}: {name} held-out accuracy={h.get('accuracy')}; generalization="
              f"{h.get('generalization')}; drawdown={h.get('drawdown')}; {audit}")
        return 0
    v = json.loads((rd / "results.json").read_text(encoding="utf-8"))["versions"][name]
    c = v["check"]
    verdict = "ERROR" if c["error"] else "PASS exact" if c["exact_pass"] else "PASS snap" if c["snap_pass"] else "FAIL"
    print(f"{rd.as_posix()}: {name} {verdict}; recovered={v['task']['recovered']}; {audit}")
    return 0


def cmd_agent_mock(args) -> int:
    from .agent.mock import mock_agent

    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else ""
    mock_agent(Path(args.blind), Path(args.program).read_text(encoding="utf-8"), notes)
    return 0


def cmd_ast(args) -> int:
    from .ast_tool import run_ast
    from .run import all_runs, is_step1

    if args.all:
        # the distill/heldout split is the step-1 protocol (seed 2 held out); dataset runs come with --split all
        runs = [rd for rd in all_runs(Path(args.root))
                if args.split == "all" or (is_step1(rd) and (rd.name == "seed2") == (args.split == "heldout"))]
    else:
        runs = [Path(r) for r in args.runs]
    if not runs:
        print("nothing to do: give run folders or --all")
        return 2
    for rd in runs:
        try:
            res = run_ast(rd)
        except (ValueError, KeyError, OSError) as e:   # one bad run (malformed dump, missing file) never stops the batch
            print(f"{rd.as_posix()}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        v = res["versions"]["ast"]
        c = v["check"]
        verdict = "ERROR" if c["error"] else "PASS exact" if c["exact_pass"] else "PASS snap" if c["snap_pass"] else "FAIL"
        route = json.loads((rd / "ast_trace.json").read_text(encoding="utf-8"))["winner"]
        print(f"{rd.as_posix()}: {verdict}; recovered={v['task']['recovered']}; "
              f"nodes={(v['size'] or {}).get('nodes')}; route={route}", flush=True)
    return 0


def _baseline_fns() -> dict:
    from .baselines.pysr_run import run_pysr
    from .baselines.regions import run_regions
    from .baselines.tree import run_tree

    return {"tree": run_tree, "regions": run_regions, "pysr": run_pysr}


def cmd_baseline(args) -> int:
    from .lang import ProgramError
    from .run import all_runs, evaluate_run

    fns = _baseline_fns()
    names = list(fns) if args.name == "all" else [args.name]
    runs = all_runs(Path(args.root)) if args.all else [Path(r) for r in args.runs]
    if not runs:
        print("nothing to do: give run folders or --all")
        return 2
    for rd in runs:
        for name in names:
            try:
                info = fns[name](rd)
                v = evaluate_run(rd)["versions"].get(name)
            except (ValueError, KeyError, OSError, ImportError, ProgramError) as e:   # one bad run never stops the batch
                print(f"{rd.as_posix()}: {name} ERROR {type(e).__name__}: {e}", flush=True)
                continue
            if v is None:
                print(f"{rd.as_posix()}: {name} no program ({info})", flush=True)
                continue
            c = v["check"]
            verdict = "ERROR" if c["error"] else "PASS exact" if c["exact_pass"] else "PASS snap" if c["snap_pass"] \
                else "FAIL"
            print(f"{rd.as_posix()}: {name} {verdict}; recovered={v['task']['recovered']}; "
                  f"nodes={(v['size'] or {}).get('nodes')}", flush=True)
    return 0


def cmd_repair(args) -> int:
    from .repair.run import evaluate_repair, run_repair_baselines
    from .run import all_runs

    runs = all_runs(Path(args.root)) if args.all else [Path(r) for r in args.runs]
    if not runs:
        print("nothing to do: give run folders or --all")
        return 2
    fn = run_repair_baselines if args.action == "baselines" else evaluate_repair
    for rd in runs:
        try:
            res = fn(rd)
        except (ValueError, KeyError, OSError) as e:   # one bad run never stops the batch
            print(f"{rd.as_posix()}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        parts = []
        for arm, r in res.items():
            h = r.get("heldout") or {}
            g = h.get("generalization")
            parts.append(f"{arm} " + (r["error"] if r.get("error") else
                                      f"gen={'n/a' if g is None else f'{g:.2f}'} draw={h.get('drawdown')}"))
        print(f"{rd.as_posix()}: " + "; ".join(parts), flush=True)
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
    r.add_argument("--bug", action="store_true", help="train on a planted-bug target (fix experiment; use --root runs_bugs)")
    r.add_argument("--root", default=str(RUNS))
    r.add_argument("--act", default="relu", choices=["relu", "sigmoid", "tanh"],
                   help="hidden activation of trained nets (non-ReLU runs skip hand nets and get formula.py)")
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
    kinds = ap.add_mutually_exclusive_group()
    kinds.add_argument("--io", action="store_true", help="input/output-only folder (samples, no dump or weights)")
    kinds.add_argument("--fix", action="store_true", help="fix experiment: decompiled program + labelled cases")
    kinds.add_argument("--fixio", action="store_true", help="fix experiment: labelled cases only")
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

    at = sub.add_parser("ast", help="refactor dumps with the deterministic heuristics tool (no LLM)")
    at.add_argument("runs", nargs="*")
    at.add_argument("--all", action="store_true")
    at.add_argument("--split", choices=["distill", "heldout", "all"], default="distill",
                    help="with --all: distill = every step-1 run except seed2 (default), heldout = step-1 seed2 "
                         "only, all = every run including dataset runs (iris)")
    at.add_argument("--root", default=str(RUNS))
    at.set_defaults(func=cmd_ast)

    bl = sub.add_parser("baseline", help="step-4 baselines: tree, regions, pysr (needs --group pysr), all")
    bl.add_argument("name", choices=["tree", "regions", "pysr", "all"])
    bl.add_argument("runs", nargs="*")
    bl.add_argument("--all", action="store_true")
    bl.add_argument("--root", default=str(RUNS))
    bl.set_defaults(func=cmd_baseline)

    rr = sub.add_parser("repair", help="fix experiment: weight-space arms and tree (baselines), or re-evaluate all arms")
    rr.add_argument("action", choices=["baselines", "evaluate"])
    rr.add_argument("runs", nargs="*")
    rr.add_argument("--all", action="store_true")
    rr.add_argument("--root", default=str(RUNS))
    rr.set_defaults(func=cmd_repair)

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
