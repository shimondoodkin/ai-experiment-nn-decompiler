"""Mock refactoring agent: writes a given program and notes, then runs the blind checker once."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def mock_agent(blind: Path, program_src: str, notes: str) -> None:
    blind = Path(blind)
    (blind / "agent.py").write_text(program_src, encoding="utf-8")
    (blind / "agent_notes.md").write_text(notes, encoding="utf-8")
    subprocess.run([sys.executable, "-I", "check.py", "agent.py"], cwd=blind, capture_output=True)
