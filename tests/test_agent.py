import json
import subprocess
import sys

from nndc.agent.collect import audit_transcripts, collect, find_transcripts, mangle
from nndc.agent.mock import mock_agent
from nndc.agent.prepare import prepare
from nndc.run import build_run


def setup_run(tmp_path):
    rd = build_run("neareq", None, root=tmp_path / "runs")
    return rd, tmp_path / "blind", tmp_path / "runs" / "blind_index.json"


def test_prepare_is_blind_and_self_contained(tmp_path):
    rd, broot, idx = setup_run(tmp_path)
    b = prepare(rd, blind_root=broot, index=idx)
    names = sorted(p.name for p in b.iterdir())
    assert names == ["INSTRUCTIONS.md", "_checker", "check.py", "dump.py", "spec.json", "weights.npz"]
    blob = "".join(p.read_text(errors="ignore") for p in b.rglob("*") if p.suffix in {".py", ".md", ".json"})
    assert "neareq" not in blob and str(rd) not in blob
    r = subprocess.run([sys.executable, "-I", "check.py", "dump.py"], cwd=b, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("PASS")
    assert (b / "agent_checks.log").read_text().count("\n") == 1


def test_blind_folder_with_spaces(tmp_path):
    rd, _, idx = setup_run(tmp_path)
    b = prepare(rd, blind_root=tmp_path / "dir with spaces", index=idx)
    assert subprocess.run([sys.executable, "-I", "check.py", "dump.py"], cwd=b).returncode == 0


def test_collect_good_and_wrong_attempts(tmp_path):
    rd, broot, idx = setup_run(tmp_path)
    heur = tmp_path / "HEURISTICS.md"
    b1 = prepare(rd, blind_root=broot, index=idx)
    mock_agent(b1, "def net(x0, x1):\n    return abs(x0 - x1) < 0.2\n",
               "Snapped.\n## Heuristics\n- relu pair -> abs\n")
    rd_, n = collect(b1, index=idx, heuristics=heur, projects=tmp_path / "noproj")
    assert rd_ == rd and n == 1
    b2 = prepare(rd, blind_root=broot, index=idx)
    mock_agent(b2, "def net(x0, x1):\n    return x0 > x1\n", "Wrong on purpose.")
    _, n2 = collect(b2, index=idx, heuristics=heur, projects=tmp_path / "noproj")
    res = json.loads((rd / "results.json").read_text())
    assert res["versions"]["agent_1"]["check"]["snap_pass"] and res["versions"]["agent_1"]["task"]["recovered"]
    assert not res["versions"][f"agent_{n2}"]["check"]["snap_pass"]
    meta = json.loads((rd / "agent_1_meta.json").read_text())
    assert meta["transcript_found"] is False
    text = heur.read_text()
    assert "relu pair -> abs" in text and '(no "## Heuristics" section found)' in text


def test_collect_same_attempt_twice_is_noop(tmp_path):
    rd, broot, idx = setup_run(tmp_path)
    b = prepare(rd, blind_root=broot, index=idx)
    mock_agent(b, "def net(x0, x1):\n    return abs(x0 - x1) < 0.2\n", "x")
    collect(b, index=idx, heuristics=tmp_path / "H.md", projects=tmp_path / "np")
    assert collect(b, index=idx, heuristics=tmp_path / "H.md", projects=tmp_path / "np")[1] is None


def _transcript(tmp_path, b):
    return [{"timestamp": "2026-10-27T10:00:00Z",
             "message": {"model": "claude-x", "usage": {"input_tokens": 100, "output_tokens": 10},
                         "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": str(b / "dump.py")}}]}},
            {"timestamp": "2026-10-27T10:01:30Z",
             "message": {"usage": {"input_tokens": 50, "output_tokens": 5},
                         "content": [{"type": "tool_use", "name": "Read",
                                      "input": {"file_path": str(tmp_path / "runs" / "secret.json")}}]}}]


def test_audit_flags_outside_paths(tmp_path):
    b = tmp_path / "blind" / "m1"
    b.mkdir(parents=True)
    proj = tmp_path / "projects" / mangle(b)
    proj.mkdir(parents=True)
    (proj / "s.jsonl").write_text("\n".join(json.dumps(line) for line in _transcript(tmp_path, b)))
    a = audit_transcripts(find_transcripts(b, projects=tmp_path / "projects"), b)
    assert a["transcript_found"] and a["model"] == "claude-x"
    assert a["duration_s"] == 90 and a["tokens"] == {"input": 150, "output": 15}
    assert a["outside_paths"] == [str(tmp_path / "runs" / "secret.json")]


def test_collect_with_explicit_transcript(tmp_path):
    # subagent runs keep their transcript elsewhere; collect accepts it explicitly
    rd, broot, idx = setup_run(tmp_path)
    b = prepare(rd, blind_root=broot, index=idx)
    mock_agent(b, "def net(x0, x1):\n    return abs(x0 - x1) < 0.2\n", "x")
    tr = tmp_path / "agent.output"
    tr.write_text("\n".join(json.dumps(line) for line in _transcript(tmp_path, b)))
    _, n = collect(b, index=idx, heuristics=tmp_path / "H.md", projects=tmp_path / "np", transcripts=[tr])
    meta = json.loads((rd / f"agent_{n}_meta.json").read_text())
    assert meta["transcript_found"] and meta["outside_paths"] and meta["model"] == "claude-x"
