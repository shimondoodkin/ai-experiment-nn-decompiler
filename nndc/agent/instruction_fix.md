# Instructions

`program.py` is a program decompiled from a neural network. It reproduces the network, including its
mistakes. `repair.csv` lists inputs with the correct label, the network's answer, and whether the
network is wrong there. Fix the algorithm so it is right on these cases. Keep it unchanged where the
network is right, and keep it short and readable. Look for the general rule behind the failing cases;
do not add special cases for single points. Afterwards, write down what you changed and why.

## Practical details

- Read `program.py`, `repair.csv` (columns: the inputs, `net` = the network's answer, `label` = the
  correct answer, `failing` = 1 where the network is wrong) and `spec.json` (inputs, their types and
  bounds, and the output kind).
- Write your version to `agent.py`. Keep `def net(x0, ...)` with exactly the same arguments as in
  `program.py`.
- Score it with `python score.py agent.py`. It reports how many cases are right, how many of the
  network's failures you fixed, and how many of its right answers you broke.
- The language your code must use: {ALLOWLIST}
- Write `agent_notes.md`: what you changed and why.
- Work only with the files in this folder.
