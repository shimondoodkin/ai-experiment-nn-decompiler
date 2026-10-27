import json
import subprocess
import sys

import numpy as np
import pytest

from nndc.agent.collect import collect
from nndc.agent.oracle import decode, encode
from nndc.agent.prepare import prepare
from nndc.check import build_input_set
from nndc.report import _best_attempt
from nndc.run import build_run, load_run


def test_oracle_roundtrip():
    data = bytes(range(256)) * 10
    enc = encode(data, "m123abc")
    assert enc != data and decode(enc, "m123abc") == data and decode(enc, "other") != data


def test_io_folder_contents_do_not_leak(tmp_path):
    rd = build_run("window", None, tmp_path / "runs")
    b = prepare(rd, blind_root=tmp_path / "blind", index=tmp_path / "idx.json", kind="io")
    names = {p.name for p in b.iterdir()}
    assert {"samples.csv", "oracle.bin", "check.py", "spec.json", "INSTRUCTIONS.md", "_checker"} <= names
    assert not names & {"dump.py", "weights.npz"}
    assert "window" not in (b / "spec.json").read_text()
    with pytest.raises(Exception):
        np.load(b / "oracle.bin")
    lines = (b / "samples.csv").read_text().splitlines()
    assert lines[0] == "x0,out,class" and len(lines) == 2001
    assert json.loads((tmp_path / "idx.json").read_text())[b.name]["kind"] == "io"


def test_io_check_runs_without_boundary_points(tmp_path):
    rd = build_run("window", None, tmp_path / "runs")
    b = prepare(rd, blind_root=tmp_path / "blind", index=tmp_path / "idx.json", kind="io")
    (b / "agent.py").write_text("def net(x0):\n    return abs(x0) < 0.5\n")
    out = subprocess.run([sys.executable, "-I", "check.py", "agent.py"], cwd=b, capture_output=True, text=True)
    assert out.stdout.startswith("PASS"), out.stdout + out.stderr
    spec, mlp = load_run(rd)
    plain, _ = build_input_set(mlp, spec, boundary=False)
    assert f"{plain.shape[0]} inputs" in out.stdout


def test_collect_io_kind_names_and_forbidden_audit(tmp_path):
    rd = build_run("window", None, tmp_path / "runs")
    idx, heur = tmp_path / "idx.json", tmp_path / "H.md"
    b = prepare(rd, blind_root=tmp_path / "blind", index=idx, kind="io")
    (b / "agent.py").write_text("def net(x0):\n    return abs(x0) < 0.5\n")
    tr = tmp_path / "t.jsonl"
    tr.write_text(json.dumps({"message": {"content": [
        {"type": "tool_use", "input": {"file_path": str(b / "oracle.bin")}}]}}) + "\n")
    rd2, n = collect(b, index=idx, heuristics=heur, transcripts=[tr])
    meta = json.loads((rd / "io_agent_1_meta.json").read_text())
    assert n == 1 and (rd / "io_agent_1.py").exists() and not (rd / "agent_1.py").exists()
    assert meta["forbidden_paths"] and not heur.exists()
    res = json.loads((rd / "results.json").read_text())
    assert res["versions"]["io_agent_1"]["check"]["snap_pass"]
    assert _best_attempt(res["versions"]) is None   # io attempts are not Claude-dump attempts


def test_cli_collect_reports_the_io_attempt(tmp_path, monkeypatch, capsys):
    import nndc.agent.prepare as prep
    from nndc.cli import main

    rd = build_run("window", None, tmp_path / "runs")
    (rd / "agent_1.py").write_text("def net(x0):\n    return x0\n")   # an older dump attempt that FAILs
    idx = tmp_path / "idx.json"
    monkeypatch.setattr(prep, "INDEX", idx)
    b = prepare(rd, blind_root=tmp_path / "blind", index=idx, kind="io")
    (b / "agent.py").write_text("def net(x0):\n    return abs(x0) < 0.5\n")
    tr = tmp_path / "t.jsonl"
    tr.write_text("")
    monkeypatch.chdir(tmp_path)   # HEURISTICS.md would be written here (it is not, for io)
    assert main(["agent", "collect", str(b), "--transcript", str(tr)]) == 0
    out = capsys.readouterr().out
    assert "io_agent_1 PASS" in out and "agent_1 FAIL" not in out
