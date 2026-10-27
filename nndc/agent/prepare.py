"""Build a blind folder for a refactoring run: no task name, no real names, no targets."""
from __future__ import annotations

import json
import secrets
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from ..lang import allowlist_summary
from ..spec import Spec

BLIND_ROOT = Path(tempfile.gettempdir()) / "nndc-blind"
INDEX = Path("runs") / "blind_index.json"
_PKG = Path(__file__).resolve().parent.parent
CHECKER_MODULES = ["spec.py", "model.py", "lang.py", "check.py"]


def load_index(index: Path) -> dict:
    return json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}


def prepare(rd: Path, blind_root: Path = BLIND_ROOT, index: Path = INDEX) -> Path:
    idx = load_index(index)
    while True:
        bid = "m" + secrets.token_hex(3)
        b = Path(blind_root) / bid
        if bid not in idx and not b.exists():
            break
    b.mkdir(parents=True)
    shutil.copyfile(rd / "dump.py", b / "dump.py")
    shutil.copyfile(rd / "weights.npz", b / "weights.npz")
    (b / "spec.json").write_text(json.dumps(Spec.load(rd / "spec.json").blind(), indent=2) + "\n", encoding="utf-8")
    instr = (_PKG / "agent" / "instruction.md").read_text(encoding="utf-8")
    (b / "INSTRUCTIONS.md").write_text(instr.replace("{ALLOWLIST}", allowlist_summary()), encoding="utf-8")
    shutil.copyfile(_PKG / "agent" / "check_template.py", b / "check.py")
    chk = b / "_checker"
    chk.mkdir()
    (chk / "__init__.py").write_text("", encoding="utf-8")
    for m in CHECKER_MODULES:
        shutil.copyfile(_PKG / m, chk / m)
    idx[bid] = {"run": Path(rd).resolve().as_posix(), "prepared": datetime.now().isoformat(timespec="seconds")}
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
    return b


def suggested_commands(blind: Path) -> list[str]:
    return [f'cd "{blind}"', 'claude "Read INSTRUCTIONS.md and follow it."']
