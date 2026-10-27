# Instructions

`repair.csv` lists inputs with the correct label, the answer of an existing system, and whether that
system is wrong there. Write a short, readable program that is right on these cases. Look for the
general rule behind the cases; do not add special cases for single points. Afterwards, write down
what you did and why.

## Practical details

- Read `repair.csv` (columns: the inputs, `net` = the existing system's answer, `label` = the correct
  answer, `failing` = 1 where the system is wrong) and `spec.json` (inputs, their types and bounds,
  and the output kind).
- Write your program to `agent.py` as `def net(x0, ...)`, with the input names from `spec.json`. For a
  binary output, return the 0/1 decision; for argmax, return the class index.
- Score it with `python score.py agent.py`. It reports how many cases are right, how many of the
  system's failures you fixed, and how many of its right answers you broke.
- The language your code must use: {ALLOWLIST}
- Write `agent_notes.md`: what you did and why.
- Work only with the files in this folder.
