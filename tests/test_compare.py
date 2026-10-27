import json

from nndc.compare import comparison_md, fix_table, loop_counts, method_table, recovery_matrix
from nndc.report import build_report
from nndc.run import build_run, evaluate_run


def _absdiff_with(tmp_path, **files):
    rd = build_run("absdiff", None, tmp_path)
    for name, text in files.items():
        (rd / name).write_text(text)
    evaluate_run(rd)
    return rd


def test_method_table_counts_and_no_program(tmp_path):
    _absdiff_with(tmp_path, **{"tree.py": "def net(x0, x1):\n    return abs(x0 - x1)\n"})
    md = method_table(tmp_path, "all")
    assert "| tree | 1/1 | 1/1 | 1/1 |" in md
    assert "| PySR | 0/1 | 0/1 | 0/1 |" in md
    pysr_row = next(r for r in md.splitlines() if r.startswith("| PySR |"))
    assert pysr_row.rstrip().endswith("| 1 |")   # no program on 1 run


def test_io_row_invalid_when_too_many_flagged(tmp_path):
    good = "def net(x0, x1):\n    return abs(x0 - x1)\n"
    for task in ("absdiff", "max"):
        rd = build_run(task, None, tmp_path)
        (rd / "io_agent_1.py").write_text(good if task == "absdiff" else "def net(x0, x1):\n    return max(x0, x1)\n")
        meta = {"forbidden_paths": ["oracle.bin"] if task == "max" else [], "outside_paths": []}
        (rd / "io_agent_1_meta.json").write_text(json.dumps(meta))
        evaluate_run(rd)
    md = method_table(tmp_path, "all")
    assert "| Claude (input/output only) | invalid (audit)" in md


def test_recovery_matrix_marks_methods(tmp_path):
    _absdiff_with(tmp_path, **{"tree.py": "def net(x0, x1):\n    return abs(x0 - x1)\n",
                               "regions.py": "def net(x0, x1):\n    return x0\n"})
    md = recovery_matrix(tmp_path)
    row = next(r for r in md.splitlines() if r.startswith("| absdiff/hand"))
    assert "✓" in row and "✗" in row and "–" in row


def test_loop_counts_from_check_logs(tmp_path):
    rd = _absdiff_with(tmp_path, **{"agent_1.py": "def net(x0, x1):\n    return abs(x0 - x1)\n"})
    (rd / "agent_1_checks.log").write_text("t\tFAIL  [x]\tagent.py\nt\tFAIL  [x]\tagent.py\nt\tPASS (snap)  [x]\tagent.py\n")
    md = loop_counts([tmp_path])
    assert "| Claude (dump) | formula | 1 | 3 | 3 | 1 | 1 | 2 |" in md


def test_fix_table_reads_repair_results(tmp_path):
    rd = build_run("xor", 0, tmp_path / "bugs", bug=True)
    (rd / "repair").mkdir()
    row = {"repair": {"efficacy": 1.0, "accuracy": 1.0}, "heldout": {"generalization": None, "drawdown": 0.0,
           "accuracy": 1.0}, "nodes": 7, "added_tests": [{"test": "x0 > 0", "support": 1, "flagged": True}],
           "reinsertion": {"exact_pass": True, "snap_pass": True, "ramps": 0, "task_match": 1.0, "box_mismatch": 0.0},
           "error": None}
    (rd / "repair" / "results.json").write_text(json.dumps({"fix_1": row}))
    md = fix_table([tmp_path / "bugs"])
    assert "| Claude fix (with program) | 1 | 1.00 | n/a | 0.00 | 1.00 |" in md and "| 1 | 1/1 |" in md


def test_report_includes_comparison(tmp_path):
    _absdiff_with(tmp_path)
    assert "## Comparison (step 4)" in build_report(tmp_path)
    assert "## Comparison (step 4)" in comparison_md(tmp_path, tmp_path, tmp_path / "none")


def test_dump_attempts_with_outside_path_flags_still_count(tmp_path):
    rd = _absdiff_with(tmp_path, **{"agent_1.py": "def net(x0, x1):\n    return abs(x0 - x1)\n",
                                    "agent_1_meta.json": json.dumps({"outside_paths": ["/len"]})})
    evaluate_run(rd)
    md = method_table(tmp_path, "all")
    assert "| Claude (dump) | 1/1 | 1/1 | 1/1 |" in md and "outside-path flag" in md


def test_iris_cv_table_from_repair_results(tmp_path):
    from nndc.compare import cv_table

    rd = tmp_path / "iris_petal" / "seed0"
    (rd / "repair").mkdir(parents=True)
    (rd / "repair" / "results.json").write_text(json.dumps(
        {"w_ft": {"cv": {"generalization": [0.4, 0.1], "drawdown": [0.01, 0.0]}}}))
    md = cv_table([tmp_path])
    assert "| fine-tune | 1 | 0.40 | 0.01 |" in md


def test_fix_table_notes_bug_learned_and_formats_cells(tmp_path):
    rd = build_run("xor", 0, tmp_path / "bugs", bug=True)
    (rd / "repair").mkdir()
    (rd / "repair" / "results.json").write_text(json.dumps({"w_lp": {"error": "infeasible"},
        "fix_1": {"repair": {"efficacy": 1.0, "n_failing": 1}, "heldout": {"accuracy": 1.0}, "nodes": 3,
                  "added_tests": [], "reinsertion": {"task_match": 1.0, "box_mismatch": 0.0}, "error": None}}))
    md = fix_table([tmp_path / "bugs"])
    assert "bug learned" in md and "xor/seed0" in md
    assert not any(line.rstrip().endswith(")") for line in md.splitlines() if line.startswith("|"))


def test_fix_table_leaves_out_nets_without_failures(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    (rd / "repair").mkdir()
    (rd / "repair" / "results.json").write_text(json.dumps(
        {"w_ft": {"repair": {"efficacy": None, "n_failing": 0}, "heldout": {"accuracy": 1.0}, "error": None}}))
    md = fix_table([tmp_path])
    assert "no failing repair cases" in md and "absdiff/hand" in md and "| fine-tune |" not in md


def test_loop_counts_treat_errors_as_not_passing(tmp_path):
    rd = _absdiff_with(tmp_path, **{"agent_1.py": "def net(x0, x1):\n    return abs(x0 - x1)\n"})
    (rd / "agent_1_checks.log").write_text("t\tERROR: x\tagent.py\nt\tFAIL  [x]\tagent.py\nt\tPASS (snap)  [x]\tagent.py\n")
    assert "| Claude (dump) | formula | 1 | 3 | 3 | 1 | 1 | 2 |" in loop_counts([tmp_path])
