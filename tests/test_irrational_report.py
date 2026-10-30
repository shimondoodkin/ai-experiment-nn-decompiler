import json

from nndc.irrational_report import build_irrational_report


def fake(base, variant, task, seed, **kw):
    rd = base / variant / task / f"seed{seed}"
    rd.mkdir(parents=True)
    a = {"task": task, "variant": variant, "seed": f"seed{seed}", "learned": True, "true_vs_net": 0.01, "family": "exp",
         "formula_found": True, "found_terms": 0, "formula_nodes": 3, "other_families": [], "route_pick": "exp",
         "ast4_nodes": 60, "ast4_pass": True, "ast5_nodes": 3, "ast5_pass": True, "ast5_route": "family",
         "formula_py_pass": True, "hinge_only": {"hinges": 5, "nodes": 25, "passed": True}, "won_on_size": True,
         "pysr_found": False, "model": {}} | kw
    (rd / "irr_analysis.json").write_text(json.dumps(a))
    return rd


def test_report_counts_and_predictions(tmp_path):
    for s in (0, 1, 2):
        fake(tmp_path, "sigmoid", "exp", s)
        fake(tmp_path, "tanh", "exp", s, formula_found=s == 0)
    fake(tmp_path, "relu", "exp", 0, learned=False, formula_found=False)
    md = build_irrational_report(tmp_path)
    assert "## Snapping to formulas" in md and "## How the nets model the functions" in md
    assert "| sigmoid | 3 | 3 (100%) |" in md                 # learned, formula found
    assert "| tanh | 3 | 1 (33%) |" in md
    assert "formula found on at least 70% of the learned development nets" in md
    assert "| 4/6 (67%) | not met |" in md                     # 3 + 1 of the 6 learned nets


def test_heldout_runs_are_separate(tmp_path):
    fake(tmp_path, "sigmoid", "exp", 0)
    fake(tmp_path, "sigmoid", "decay", 0, family="exp")
    (tmp_path / "sigmoid" / "heldout.json").write_text(json.dumps({"version": "ast5", "freeze": "abc",
                                                                   "runs": ["decay/seed0"]}))
    md = build_irrational_report(tmp_path)
    assert "## Held-out" in md and "freeze abc" in md
    dev = md[md.index("## Snapping to formulas"):md.index("## Held-out")]
    assert "| sigmoid | 1 | 1 (100%) |" in dev
