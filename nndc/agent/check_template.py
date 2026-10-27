"""Check a program against the network in this folder.

Usage: python check.py agent.py
"""
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # also under `python -I`, which does not add the script folder

from _checker.check import CheckResult, check_program, format_result  # noqa: E402
from _checker.lang import ProgramError, read_program  # noqa: E402
from _checker.model import load_npz  # noqa: E402
from _checker.spec import Spec  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python check.py <program.py>")
        return 2
    program = Path(argv[0])
    spec = Spec.load(HERE / "spec.json")
    mlp = load_npz(HERE / "weights.npz")
    try:
        r = check_program(read_program(program), mlp, spec)
    except ProgramError as e:
        r = CheckResult(False, False, "empirical", 0, error=str(e))
    text = format_result(r, spec)
    print(text, end="")
    with open(HERE / "agent_checks.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{text.splitlines()[0]}\t{program.name}\n")
    return 0 if r.snap_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
