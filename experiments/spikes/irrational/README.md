# Spike: irrational functions (2026-10-30)

Can a small net's model of exp, log, sqrt, 1/x, x², a circle, a distance, a bump or a product be snapped back to the
formula? It led to the irrational-functions project and `ast5` (`docs/design/2026-10-30-irrational-functions-design.md`).

- `spike.py [task ...]` trains seed 0 of each task as ReLU, sigmoid and wide ReLU (4× hidden, 3× steps). For each net
  it records how the net models the curve (kink or unit-centre positions, the circle's boundary), fits families
  `a·f(b·x + c) + d` to the net, snaps them, checks them, and adds greedy hinge terms in 1-D. It also runs `ast4`.
  It writes `out/<task>_<act>.json` (git-ignored); it skips tasks that already have one. 2-D sigmoid nets take about
  10 minutes each, mostly `ast4`.
- `summary.py` prints one line per net from `out/`.

It ran from the repo root with the code of `da65bb0` (before `exp`/`log`/`sqrt` were in the language). The results
table is in the design's "Why" section. Its 2-D families assume the centre is at the origin; `ast5` does not.
