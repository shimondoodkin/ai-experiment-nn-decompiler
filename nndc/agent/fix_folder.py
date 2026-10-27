"""Fill a blind folder for the fix experiment: the decompiled program (kind `fix` only), the labelled
repair cases with the network's answers, a scorer, and the instruction. No task name, no weights."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..repair.data import repair_data
from ..repair.metrics import decide, right
from ..run import load_run
from ..spec import Spec

_PKG = Path(__file__).resolve().parent.parent


def fill_fix_folder(b: Path, rd: Path, spec: Spec, kind: str, program: Path | None) -> None:
    from ..lang import allowlist_summary
    from ..repair.run import _tol, faithful_program

    _, mlp = load_run(rd)
    data = repair_data(rd)
    tol = _tol(spec, data)
    X, y = data.X_repair, data.y_repair
    net = decide(spec, mlp.forward(X.astype(np.float32)))
    failing = ~right(spec, net, y, tol)
    disc = [i.type != "float" for i in spec.inputs]
    lines = [",".join(spec.arg_names + ["net", "label", "failing"])]
    for x, n, lab, f in zip(X, net, y, failing):
        cells = [str(int(v)) if d else repr(float(v)) for v, d in zip(x, disc)]
        fmt = (lambda v: repr(float(v))) if spec.output == "regression" else (lambda v: str(int(v)))  # noqa: E731
        lines.append(",".join(cells + [fmt(n), fmt(lab), str(int(f))]))
    (b / "repair.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (b / "score.json").write_text(json.dumps({"tolerance": tol}) + "\n", encoding="utf-8")
    if kind == "fix":
        shutil.copyfile(program if program is not None else faithful_program(rd), b / "program.py")
    name = "instruction_fix.md" if kind == "fix" else "instruction_fixio.md"
    instr = (_PKG / "agent" / name).read_text(encoding="utf-8")
    (b / "INSTRUCTIONS.md").write_text(instr.replace("{ALLOWLIST}", allowlist_summary()), encoding="utf-8")
    shutil.copyfile(_PKG / "agent" / "score_template.py", b / "score.py")
    chk = b / "_checker"
    chk.mkdir()
    (chk / "__init__.py").write_text("", encoding="utf-8")
    for m in ("spec.py", "lang.py"):
        shutil.copyfile(_PKG / m, chk / m)
