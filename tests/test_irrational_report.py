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


def test_near_formula_column_and_explain_link(tmp_path):
    fake(tmp_path, "sigmoid", "dist", 0, family="sqrt", formula_found=False, found_terms=None, near_family="sqrt",
         near_err=0.025, near_terms=0, near_family_is_task_family=True, explain=True)
    md = build_irrational_report(tmp_path)
    assert "near formula" in md and "| sqrt 2.5% |" in md
    assert "[explain](sigmoid/dist/seed0/explain.md)" in md


def test_gate_view_predictions(tmp_path):
    for s in (0, 1, 2):
        fake(tmp_path, "sigmoid", "dist", s, family="sqrt", formula_found=s == 0, found_terms=0 if s == 0 else None,
             near_family="sqrt" if s else None, near_err=0.025 if s == 1 else 0.05 if s == 2 else None,
             near_family_is_task_family=True if s else None, ast5_nodes=10)
    fake(tmp_path, "sigmoid", "circle", 0, family="conic", formula_found=False)    # binary: not counted in the 90% line
    md = build_irrational_report(tmp_path)
    assert "the formula is found on at least 2 of the 3 sigmoid dist nets" in md and "| 1/3 (33%) | not met |" in md
    assert "| 2/3 (67%) | not met |" in md          # found (seed0) or near of the family within 3% (seed1)
    assert "no saved list" in md
    (tmp_path / "before_minimax.json").write_text(json.dumps({"runs": {
        "sigmoid/dist/seed0": {"formula_found": True, "ast5_nodes": 12, "ast5_pass": True},
        "sigmoid/dist/seed1": {"formula_found": False, "ast5_nodes": 400, "ast5_pass": True}}}))
    md = build_irrational_report(tmp_path)
    assert "| found 1 before, 1 now; larger on 0 of 2 | met |" in md
    (tmp_path / "before_minimax.json").write_text(json.dumps({"runs": {
        "tanh/dist/seed9": {"formula_found": True, "ast5_nodes": 12, "ast5_pass": True}}}))
    assert "| found 0 before, 0 now; larger on 0 of 0 | – |" in build_irrational_report(tmp_path)   # nothing compared


def test_report_reproduction_section(tmp_path):
    fake(tmp_path, "relu", "square", 0, agent_pass=True, agent_nodes=41, ast5_nodes=55, agent_formula=False,
         ast5_formula=False, reproduced=False)
    fake(tmp_path, "relu", "square", 1, agent_pass=True, agent_nodes=12, ast5_nodes=7, agent_formula=True,
         ast5_formula=True, reproduced=True)
    fake(tmp_path, "relu", "square", 2, reproduced=None)                 # no agent attempt: not counted
    md = build_irrational_report(tmp_path)
    sec = md[md.index("## Reproduction"):]
    assert "| relu | 2 | 1/2 (50%) | 1 | 0 | 1 |" in sec               # agents passing, reproduced, smaller/equal/larger
    assert "relu/square/seed0" in sec and "55" in sec and "41" in sec and "larger" in sec
    assert "Before the distilled changes" not in sec
    (tmp_path / "before_distill.json").write_text(json.dumps({"runs": {
        "relu/square/seed0": {"reproduced": False}, "relu/square/seed1": {"reproduced": False}}}))
    assert "Before the distilled changes: 0/2" in build_irrational_report(tmp_path)


def test_heldout_reproduction_section(tmp_path):
    fake(tmp_path, "relu", "square", 0, agent_pass=True, agent_nodes=12, ast5_nodes=7, agent_formula=True,
         ast5_formula=True, reproduced=True)
    fake(tmp_path, "relu", "decay", 0, family="exp", agent_pass=True, agent_nodes=8, ast5_nodes=30,
         agent_formula=True, ast5_formula=False, reproduced=False)
    (tmp_path / "relu" / "heldout.json").write_text(json.dumps({"version": "ast5", "freeze": "abc",
                                                                "runs": ["decay/seed0"]}))
    (tmp_path / "before_distill.json").write_text(json.dumps({"runs": {"relu/square/seed0": {"reproduced": False}}}))
    md = build_irrational_report(tmp_path)
    dev = md[md.index("## Reproduction\n"):md.index("## Held-out")]
    assert "| relu | 1 | 1/1 (100%) | 1 | 0 | 0 |" in dev and "decay" not in dev
    held = md[md.index("## Reproduction (held-out)"):]
    assert "| relu | 1 | 0/1 (0%) | 0 | 0 | 1 |" in held
    assert "relu/decay/seed0: ast5 30 nodes, agent 8 (larger, not the family)" in held
    assert "Before the distilled changes" not in held
