import json

from nndc.ast_tool import run_ast
from nndc.cli import main
from nndc.report import build_report, reproduction
from nndc.run import build_run


def test_run_ast_adds_version_and_keeps_agents(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    res = run_ast(rd)
    assert (rd / "ast.py").exists() and (rd / "ast_trace.json").exists()
    assert {"dump", "agent_1", "ast"} <= set(res["versions"]) and res["versions"]["ast"]["task"]["recovered"]
    run_ast(rd)                                   # rerun: overwrites ast, keeps agent_1
    assert (rd / "agent_1.py").exists()


def _fake_run(root, seed_dir):
    rd = root / "absdiff" / seed_dir
    rd.mkdir(parents=True)
    (rd / "results.json").write_text(json.dumps({"run": f"absdiff/{seed_dir}", "versions": {
        "dump": {"size": {"nodes": 50}, "check": {"snap_pass": True, "exact_pass": True},
                 "task": {"recovered": False}, "meta": None},
        "agent_1": {"size": {"nodes": 10}, "check": {"snap_pass": True, "exact_pass": True},
                    "task": {"recovered": True}, "meta": {"outside_paths": []}},
        "ast": {"size": {"nodes": 14}, "check": {"snap_pass": True, "exact_pass": True},
                "task": {"recovered": True}, "meta": None}}}))
    (rd / "spec.json").write_text("{}")


def test_reproduction_rows_and_split(tmp_path):
    for seed_dir in ["hand", "seed2"]:
        _fake_run(tmp_path, seed_dir)
    rows = {r["run"]: r for r in reproduction(tmp_path)}
    assert rows["absdiff/seed2"]["split"] == "heldout" and rows["absdiff/hand"]["split"] == "distill"
    assert rows["absdiff/hand"]["reproduces"]          # 14 <= max(15, 15)


def test_report_has_reproduction_section_with_freeze_line(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    run_ast(rd)
    (tmp_path / "ast_freeze.txt").write_text("abc1234\n")
    md = build_report(tmp_path)
    sec = md.split("## Reproduction (step 3)")[1].split("\n## ")[0]
    assert "Tool frozen at abc1234" in sec and "absdiff/hand" in sec and "in-sample" in sec


def test_cli_ast_default_split_skips_seed2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_run("absdiff", None, root=tmp_path / "runs")
    build_run("absdiff", 2, root=tmp_path / "runs")      # a seed-2 run that must not be touched
    assert main(["ast", "--all"]) == 0
    assert (tmp_path / "runs/absdiff/hand/ast.py").exists()
    assert not (tmp_path / "runs/absdiff/seed2/ast.py").exists()
