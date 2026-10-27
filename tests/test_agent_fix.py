import json
import subprocess
import sys

from nndc.agent.collect import collect
from nndc.agent.prepare import prepare
from nndc.run import build_run


def _bug_xor(tmp_path):
    rd = build_run("xor", 0, tmp_path / "runs", bug=True)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return x0 != x1 or (x0 == 1 and x1 == 1)\n")
    return rd


def test_fix_folder_and_score(tmp_path):
    rd = _bug_xor(tmp_path)
    b = prepare(rd, blind_root=tmp_path / "b", index=tmp_path / "i.json", kind="fix", program=rd / "agent_1.py")
    assert (b / "program.py").read_text() == (rd / "agent_1.py").read_text()
    assert "xor" not in (b / "spec.json").read_text() and "xor" not in (b / "INSTRUCTIONS.md").read_text()
    head = (b / "repair.csv").read_text().splitlines()[0]
    assert head == "x0,x1,net,label,failing"
    (b / "agent.py").write_text("def net(x0, x1):\n    return x0 != x1\n")
    out = subprocess.run([sys.executable, "-I", "score.py", "agent.py"], cwd=b, capture_output=True, text=True)
    assert out.stdout.splitlines() == ["repair set: 4 points; your program is right on 4",
                                       "of the 1 points the network gets wrong, now right: 1",
                                       "of the 3 points the network gets right, now wrong: 0"], out.stdout + out.stderr
    assert (b / "agent_checks.log").read_text().count("\n") == 1


def test_fixio_folder_has_no_program(tmp_path):
    rd = _bug_xor(tmp_path)
    b = prepare(rd, blind_root=tmp_path / "b", index=tmp_path / "i.json", kind="fixio")
    assert not (b / "program.py").exists() and (b / "repair.csv").exists()
    assert "program.py" not in (b / "INSTRUCTIONS.md").read_text()


def test_collect_fix_runs_repair_evaluation(tmp_path):
    rd = _bug_xor(tmp_path)
    idx = tmp_path / "i.json"
    b = prepare(rd, blind_root=tmp_path / "b", index=idx, kind="fix")   # program defaults to the faithful one
    assert (b / "program.py").exists()
    (b / "agent.py").write_text("def net(x0, x1):\n    return x0 != x1\n")
    _, n = collect(b, index=idx, heuristics=tmp_path / "H.md", transcripts=[])
    res = json.loads((rd / "repair" / "results.json").read_text())
    assert n == 1 and (rd / "fix_1.py").exists() and res["fix_1"]["heldout"]["accuracy"] == 1.0
    assert "fix_1" in (rd / "run_git.txt").read_text()
