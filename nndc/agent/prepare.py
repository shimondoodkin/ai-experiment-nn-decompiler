"""Build a blind folder for an agent run: no task name, no real names, no targets.

Kinds: `agent` (refactor the dump, step 2), `io` (reproduce a black box from samples, step-4 B5),
`fix` / `fixio` (fix a decompiled program from failing task cases, step-4 experiment R).
"""
from __future__ import annotations

import json
import secrets
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from ..lang import allowlist_summary
from ..run import load_run
from ..spec import Spec
from .oracle import encode

BLIND_ROOT = Path(tempfile.gettempdir()) / "nndc-blind"
INDEX = Path("runs") / "blind_index.json"
_PKG = Path(__file__).resolve().parent.parent
CHECKER_MODULES = ["spec.py", "model.py", "lang.py", "check.py"]
KINDS = {"agent": "agent", "io": "io_agent", "fix": "fix", "fixio": "fixio"}   # kind -> file prefix in the run
N_IO_SAMPLES = 2000


def load_index(index: Path) -> dict:
    return json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}


def _num(v: float, discrete: bool) -> str:
    return str(int(v)) if discrete else repr(float(v))


def _samples_csv(rd: Path, spec: Spec) -> str:
    from ..baselines.common import net_samples

    _, mlp = load_run(rd)
    X, _, dec = net_samples(spec, mlp, n=N_IO_SAMPLES)
    out = mlp.forward(X.astype(np.float32)).astype(np.float64)
    disc = [i.type != "float" for i in spec.inputs]
    head = spec.arg_names + (["out"] if out.shape[1] == 1 else [f"out{k}" for k in range(out.shape[1])])
    head += [] if spec.output == "regression" else ["class"]
    rows = [",".join(head)]
    for x, o, c in zip(X, out, dec):
        cells = [_num(v, d) for v, d in zip(x, disc)] + [repr(float(v)) for v in o]
        cells += [] if spec.output == "regression" else [str(int(c))]
        rows.append(",".join(cells))
    return "\n".join(rows) + "\n"


def _copy_checker(b: Path, modules: list[str]) -> None:
    chk = b / "_checker"
    chk.mkdir()
    (chk / "__init__.py").write_text("", encoding="utf-8")
    for m in modules:
        src = _PKG / m if (_PKG / m).exists() else _PKG / "agent" / m
        shutil.copyfile(src, chk / m)


def _instructions(b: Path, name: str) -> None:
    instr = (_PKG / "agent" / name).read_text(encoding="utf-8")
    (b / "INSTRUCTIONS.md").write_text(instr.replace("{ALLOWLIST}", allowlist_summary()), encoding="utf-8")


def prepare(rd: Path, blind_root: Path = BLIND_ROOT, index: Path = INDEX, kind: str = "agent",
            program: Path | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; known: {', '.join(KINDS)}")
    idx = load_index(index)
    while True:
        bid = "m" + secrets.token_hex(3)
        b = Path(blind_root) / bid
        if bid not in idx and not b.exists():
            break
    b.mkdir(parents=True)
    blind = Spec.load(rd / "spec.json").blind()
    (b / "spec.json").write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
    spec = Spec.from_json(blind)
    if kind == "agent":
        shutil.copyfile(rd / "dump.py", b / "dump.py")
        shutil.copyfile(rd / "weights.npz", b / "weights.npz")
        _instructions(b, "instruction.md")
        (b / "scratch").mkdir()   # the agent's own scratch space (parallel agents shared the session scratchpad)
        shutil.copyfile(_PKG / "agent" / "check_template.py", b / "check.py")
        _copy_checker(b, CHECKER_MODULES)
    elif kind == "io":
        (b / "samples.csv").write_text(_samples_csv(rd, spec), encoding="utf-8")
        (b / "oracle.bin").write_bytes(encode((rd / "weights.npz").read_bytes(), bid))
        _instructions(b, "instruction_io.md")
        shutil.copyfile(_PKG / "agent" / "check_io_template.py", b / "check.py")
        _copy_checker(b, CHECKER_MODULES + ["oracle.py"])
    else:
        from .fix_folder import fill_fix_folder

        fill_fix_folder(b, rd, spec, kind, program)
    idx[bid] = {"run": Path(rd).resolve().as_posix(), "kind": kind,
                "prepared": datetime.now().isoformat(timespec="seconds")}
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
    return b


def suggested_commands(blind: Path) -> list[str]:
    return [f'cd "{blind}"', 'claude "Read INSTRUCTIONS.md and follow it."']
