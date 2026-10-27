"""Score a program on the labelled cases in repair.csv.

Usage: python score.py agent.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # also under `python -I`, which does not add the script folder

from _checker.lang import ProgramError, evaluate, parse_program, read_program  # noqa: E402
from _checker.spec import Spec  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python score.py <program.py>")
        return 2
    program = Path(argv[0])
    spec = Spec.load(HERE / "spec.json")
    tol = json.loads((HERE / "score.json").read_text(encoding="utf-8"))["tolerance"]
    with open(HERE / "repair.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[n]) for n in spec.arg_names] for r in rows])
    net = np.array([float(r["net"]) for r in rows])
    label = np.array([float(r["label"]) for r in rows])
    try:
        raw = evaluate(parse_program(read_program(program), spec.arg_names), X)
    except ProgramError as e:
        text = f"ERROR: {e}\n"
    else:
        pred = (raw > spec.threshold).astype(float) if spec.output == "binary" else raw

        def right(p):
            return np.abs(p - label) <= tol if spec.output == "regression" else p == label

        ok_net, ok_new = right(net), right(pred)
        text = (f"repair set: {len(X)} points; your program is right on {int(ok_new.sum())}\n"
                f"of the {int((~ok_net).sum())} points the network gets wrong, now right: "
                f"{int((ok_new & ~ok_net).sum())}\n"
                f"of the {int(ok_net.sum())} points the network gets right, now wrong: {int((~ok_new & ok_net).sum())}\n")
    print(text, end="")
    with open(HERE / "agent_checks.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{text.splitlines()[0]}\t{program.name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
