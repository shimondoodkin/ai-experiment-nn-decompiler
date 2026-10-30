# Iris with sigmoid nets

This file is written by hand; `runs_sigmoid/REPORT.md` holds the full tables.

Nets were trained with the same sizes, seeds and settings as the ReLU Iris runs, but with sigmoid hidden units:
- `iris/`: all 4 measurements;
- `iris_petal/`: petal measurements only.

There is no known formula, so success means a short program that passes the checker against the net.

**AST tool split.** Development used seeds 0–2. Held-out seeds 3–5 were run once, with the frozen tool (freeze
`15b943b`; the held-out runs record `797e0ae`, a data-only commit on top of the freeze, so the code is identical).

| run | split | net learned | dump nodes | Claude agent | AST tool | tool route |
|---|---|---|---|---|---|---|
| iris/seed0 | development | yes | 478 | PASS snap, 81 | PASS exact, 478 (the dump) | structural-exact |
| iris/seed1 | development | yes | 480 | PASS snap, 57 | PASS exact, 480 (the dump) | structural-exact |
| iris/seed2 | development | yes | 478 | PASS snap, 73 | PASS snap, 72 | linear-scores |
| iris/seed3 | held-out | yes | 476 | PASS snap, 70 | PASS snap, 74 | linear-scores |
| iris/seed4 | held-out | yes | 482 | PASS snap, 76 | PASS snap, 74 | linear-scores |
| iris/seed5 | held-out | yes | 478 | PASS snap, 70 | PASS snap, 74 | linear-scores |
| iris_petal/seed0 | development | yes | 374 | PASS snap, 48 | PASS snap, 42 | linear-scores |
| iris_petal/seed1 | development | no | 372 | PASS snap, 42 | PASS snap, 42 | linear-scores |
| iris_petal/seed2 | development | yes | 366 | PASS snap, 42 | PASS snap, 42 | linear-scores |
| iris_petal/seed3 | held-out | yes | 370 | PASS snap, 42 | PASS snap, 42 | linear-scores |
| iris_petal/seed4 | held-out | yes | 374 | PASS snap, 48 | PASS snap, 42 | linear-scores |
| iris_petal/seed5 | held-out | no | 368 | PASS snap, 42 | PASS snap, 42 | linear-scores |

"Net learned" is no for two Iris-petal nets: their accuracy was 0.946 and 0.945, against a bar of 0.95. They stay
in, as the ReLU nets that did not learn did.

**What it shows**
- **Tool vs agents on held-out nets:** all 6 held-out nets get a passing linear rule from the frozen tool, about as
  short as the agents' rules.
- **Development seeds 0–1 (4 measurements):** the tool fell back to the dump.
  - Both linear routes miss by a few confident points: the cascade is 16 points off on seed 0 and 1 on seed 1.
  - Their fit is a logistic regression. The agents used max-margin fits on confident points, and on seed 0 an
    `x3²` term, because that boundary is curved.
  - The tool was not tuned after seeing this, as the freeze rule requires.
- **What the agents wrote:** two-cut decision lists (`if s_a > 0: a; if s_c > 0: c; else b`), the same shape as for
  the ReLU nets. The classes form ordered bands, and classes 0 and 2 never border each other.

**Later (2026-10-29):** `ast2` (the kink-merge tool) replaced `ast` in every sigmoid run. Kink merge does not apply to
argmax nets, so on Iris `ast2` is the same program as the AST tool column above; the old `ast.py` files are in git
history.

**Later (2026-10-30):** `ast3` and then `ast4` ran on every Iris net. Kink merge does not apply to argmax nets, so
both give the same program as `ast2` here.
