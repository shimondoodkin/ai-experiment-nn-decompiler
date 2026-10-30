# Iris with tanh nets

This file is written by hand. `runs_tanh/REPORT.md` holds the full tables.

The nets were trained with the same sizes, seeds and settings as the sigmoid and ReLU Iris runs, but with tanh hidden
units:
- `iris/`: all 4 measurements;
- `iris_petal/`: the petal measurements only.

There is no known formula, so success means a short program that passes the checker against the net.

**Tool split.** The tool is `ast3`. It rewrites tanh layers as sigmoid layers for its routes and checks every program
against the original tanh net.
- **Development:** seeds 0–2, built and studied before the freeze (`38b1e15`).
- **Held-out:** seeds 3–5, built after the freeze and run once (`runs_tanh/heldout.json`).

| run | split | net learned | dump nodes | Claude agent | `ast3` | `ast3` route |
|---|---|---|---|---|---|---|
| iris/seed0 | development | yes | 482 | PASS snap, 139 | PASS exact, 482 (the dump) | structural-exact |
| iris/seed1 | development | yes | 484 | PASS snap, 145 | PASS exact, 484 (the dump) | structural-exact |
| iris/seed2 | development | yes | 478 | PASS snap, 200 | PASS snap, 440 | sigmoid-replace |
| iris/seed3 | held-out | yes | 482 | PASS snap, 310 | PASS snap, 434 | sigmoid-replace |
| iris/seed4 | held-out | yes | 482 | PASS snap, 93 | PASS snap, 441 | sigmoid-replace |
| iris/seed5 | held-out | yes | 476 | PASS snap, 216 | PASS snap, 435 | sigmoid-replace |
| iris_petal/seed0 | development | yes | 374 | PASS snap, 42 | PASS snap, 42 | linear-scores |
| iris_petal/seed1 | development | yes | 370 | PASS snap, 42 | PASS snap, 42 | linear-scores |
| iris_petal/seed2 | development | no | 366 | PASS snap, 45 | PASS snap, 292 | sigmoid-replace |
| iris_petal/seed3 | held-out | yes | 372 | PASS snap, 42 | PASS snap, 42 | linear-scores |
| iris_petal/seed4 | held-out | no | 372 | PASS snap, 39 | PASS snap, 42 | linear-scores |
| iris_petal/seed5 | held-out | yes | 368 | PASS snap, 39 | PASS snap, 42 | linear-scores |

"Net learned" is no for two Iris-petal nets: their accuracy was 0.949 and 0.946, against a bar of 0.95. They stay in.

**What it shows**
- **Iris-petal (2 inputs) is as easy as with sigmoid.** The tool's linear scores (42 nodes) pass on 5 of 6 nets. The
  agents wrote the same two-cut decision lists (39–45 nodes).
- **4-feature Iris is harder with tanh than with sigmoid.**
  - With sigmoid, a linear route passed on 4 of 6 nets (seeds 2–5, 72–74 nodes; seeds 0–1 fell back to the dump).
  - With tanh, no linear route passes. The tool falls back to the dump or to sigmoid-replace (434–441 nodes).
  - The agents explain why: the tanh nets' boundaries bend near the edges of the input box. Single planes fail on a few
    confident points, so they used two planes per class, a plane plus a squared term, or a smaller tanh net. Their
    programs are 93–310 nodes, still about 1.4–5 times shorter than the tool's.
- **Agents guessed "Iris".** Several agents guessed the dataset from the input ranges alone (their notes say so). No
  task information was in their folders, and their programs use none.
- **One agent tried to kill every Python process** on the machine with `taskkill` to stop its own background jobs.
  The permission rules blocked it; it used its own task tools instead.

**Later (2026-10-30):** `ast4` (`docs/design/2026-10-30-ast4-design.md`) ran on every Iris net. It changes only the
kink-merge route, which does not apply to argmax nets, so its programs are the same as `ast3`'s on all 12 nets.

The `ast3` files were removed on 2026-10-30 (they are in git history). The same programs are in each run's `ast4.py`.
