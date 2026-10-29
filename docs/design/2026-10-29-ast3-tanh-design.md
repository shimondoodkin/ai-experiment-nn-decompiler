# ast3 and tanh nets

Status: approved (the user delegated the open items), 2026-10-29. Follows kink merge (`2026-10-29-kink-merge-design.md`, merged to main).

## Why

After the kink-merge merge, three things were still open (`CONCLUSIONS.md`, "Not done yet"). The user said "you can
do what is still open":
1. **The final review's tool changes.** They change programs, so they need a new tool version, a new freeze and a
   fresh held-out set:
   - a line whose normal is near an axis can be split in two;
   - the unrounded candidate keeps lines that left the box;
   - the position tolerance uses the largest input range, not the line's own range;
   - the constant is written at full precision.
2. **Two-layer sigmoid nets (median3).**
3. **Tanh nets.**

**The median3 spike (2026-10-29, scratchpad only):**
- Expanding both sigmoid layers of the median3 nets into 17-piece chords stays within 1.6–1.7% of the net.
- But the first layer's 256 kinks still form 147–224 lines after merging, pointing in 9–14 directions. Nothing
  cancels at the first layer, because the cancellation only happens after the second layer mixes the units.
- So line merging does not carry over to two layers. Median3 needs a different idea, and is left out here. The
  finding goes to `CONCLUSIONS.md`.

**Tanh** is a scaled sigmoid: `tanh(z) = 2·sigmoid(2z) − 1`. A tanh net can be rewritten exactly as a sigmoid net,
so every sigmoid move, kink merge included, applies to it.

## Goals

1. **A new tool version, `ast3`**, written as `ast3.py`, `ast3_trace.json` and `counterexamples_ast3.json`. It has
   every route of `ast2` plus:
   1. **Merge across direction:** two kinks lie on one line when `|cos| > 0.995` and their offsets match once
      both are oriented the same way. A flipped kink keeps its strength; the `+ y` part goes to the affine part,
      which `refine` refits anyway.
   2. **Fold out-of-box lines at every level:** the unrounded candidate becomes snap level `tol = 0`. That is the
      same code path as the other levels: fold lines that left the box, refit with the LP, and keep the values
      unrounded.
   3. **The position tolerance follows the line:** `t` is snapped within `tol · Σ|n_i|·span_i`, the range of
      `n·x` over the box, instead of `tol · max span`.
   4. **A rounded constant:** at each snap level the constant becomes `nicest_in(c − tol·R, c + tol·R)`, where `R`
      is the range of the net's output on the fit samples. At `tol = 0` it stays unrounded.
   5. **Tanh nets:** `refactor` rewrites a net whose hidden layers include tanh into the equivalent sigmoid net.
      The routes run on that net; the checker still compares every candidate with the original tanh net.

   Plain `nndc ast` stays the project B tool. `--name ast2` is no longer offered: its code is in git at `93d7199`,
   and running it under today's code would silently differ. `--name ast3` is the new tool.
2. **Tanh nets for every trained task,** like project B. Formula tasks xor, window, neareq, absdiff, median3 and
   tax, seeds 0–2; Iris and Iris-petal, seeds 0–5. That is 30 nets in `runs_tanh/`, with `formula.py` for the
   formula tasks. Build them with `nndc run <task> --seed <k> --act tanh --root runs_tanh`, in the order set in
   Decisions.
3. **Blind agents on all 30 tanh nets,** as in project B: two waves, each agent with its `scratch/` folder.
   - Heuristics go to `runs_tanh/HEURISTICS.md` (development) and `runs_tanh/HEURISTICS_heldout.md`.
   - The blind index is `runs_tanh/blind_index.json`.
4. **Development and freeze:**
   - **Development set:**
     - sigmoid seeds 0–5 of xor, window, neareq, absdiff and tax. Seeds 3–5 were `ast2`'s held-out set; for
       `ast3` they are development;
     - tanh formula seeds 0–1 and tanh Iris seeds 0–2.
   - The parameters in Goal 1 are fixed now. Development may fix bugs only, then comes the freeze commit.
5. **Fresh held-out sets,** used once with the frozen tool:
   - **sigmoid:** seeds 6–8 of xor, window, neareq, absdiff and tax. That is 15 new nets, built after the
     freeze, with `formula.py` and no agents;
   - **tanh:** formula seed 2 and Iris seeds 3–5. Their agents run as in Goal 3, but tool development does not
     read their notes.
6. **Replace `ast2` with `ast3`:**
   - **Sigmoid runs:** `ast3` runs on every sigmoid run, seeds 0–8.
   - **Before removing anything:** compare `ast3` with `ast2` on seeds 0–5, where both exist. The report gets
     this as a table: nodes, verdict, recovered.
   - **The removal:** once that comparison shows `ast3` is never worse, `ast2.py`, `ast2_trace.json` and
     `counterexamples_ast2.json` are removed from the sigmoid runs, in a commit of their own (CLAUDE.md:
     superseded results may go). If `ast3` is worse anywhere, nothing is removed and the report says where.
7. **Reports:**
   - `runs_sigmoid/REPORT.md` shows `ast3`. The kink section is about the newest version, with a short `ast2` vs
     `ast3` comparison on seeds 0–5.
   - `runs_tanh/REPORT.md` has the same sections, compared with `runs_sigmoid` (same tasks and seeds):
     `nndc report --root runs_tanh --out runs_tanh/REPORT.md --compare-to runs_sigmoid`. The comparison heading
     stops saying "ReLU" when the other root is not ReLU.
   - A short hand-written `runs_tanh/iris/REPORT.md`, as for sigmoid.
8. **Afterwards:** a tanh and `ast3` part in `CONCLUSIONS.md`, including the median3 finding.

## Predictions (written before any tool code)

- **The four fixes change little on these tasks.**
  - Fixes 1 and 3 only matter for normals near an axis or inputs with mixed ranges, and all current multi-input
    tasks have equal ranges.
  - Fix 4 rounds the constant. So more kink-merge programs pass at a snapped level, and fewer full-precision
    numbers appear.
  - On held-out sigmoid seeds 6–8, I expect the same route winners as `ast2` had on seeds 3–5: neareq kink merge,
    tax knot-fit, xor and window as before, absdiff mixed.
- **Tanh nets behave like sigmoid nets of twice the steepness.**
  - xor and window: recovered by agents and the tool.
  - absdiff and tax: rounded corners, faithful programs.
  - Iris: linear scores.

## Non-goals

- Two-layer nets (median3) for kink merge: see the spike above.
- Smooth functions (`exp`, `sqrt`, circles): the user's next idea, a project of its own.
- Re-running `ast2` on new nets, and changing checker tolerances.
- Step-4 baselines on tanh nets.

## Anti-goals

- **Any change to ReLU results:** the equivalence script must report 0 differences.
- **Changing an existing verdict:** `ast3` has its own counterexample file.
- **Tuning after a held-out result**, or building held-out nets before the freeze.
- **Removing `ast2` before the comparison shows `ast3` is never worse.**

## Constraints

Everything in CLAUDE.md: discipline, data commits apart from code commits, no destructive commands on the repo, and
blind-folder modules stdlib + NumPy only.

## Decisions (my calls; the user delegated these)

- **One tool version, not one per fix.** The review fixes and tanh support share one freeze and one held-out set.
  That costs one round of held-out nets instead of two.
- **No agents on sigmoid seeds 6–8.** The sigmoid agent experiment already has 30 nets. These nets test the tool,
  and the tanh nets get agents. So the `ast3` reproduction count on sigmoid held-out nets will be empty.
- **Tanh as a rewrite, not new moves.** It is exact maths, and it reuses everything. Programs on tanh nets will
  say `sigmoid(...)` where a unit stays soft.
- **Tanh held-out set as in project B** (formula seed 2 and Iris 3–5), so the two activations compare one to one.
- **Median3 left out** after the spike. A two-layer idea is future work.

- **"`ast3` is worse than `ast2`"** (Goal 6, fixed before any result). On one run, `ast3` is worse when:
  - `ast2` passes and `ast3` does not;
  - or `ast2` recovers the formula and `ast3` does not;
  - or both pass and `ast3` has more nodes.

  A change from exact to snap with fewer nodes is not "worse": the tool already prefers fewer nodes. Each verdict
  is taken as recorded, each version on the shared counterexamples plus its own. The comparison covers every
  sigmoid run with `ast2`: 45 runs, seeds 0–5 of the five tasks plus median3 and Iris. So `ast3` also runs on
  median3 and Iris.
- **Tanh build order is strict.** Development nets are built first: formula seeds 0–1, Iris and Iris-petal seeds
  0–2. Held-out nets are built only after the freeze: formula seed 2, Iris and Iris-petal seeds 3–5. So the tanh
  agents run in two waves: 18 development nets, then 12 held-out nets.
- **Tanh runs get only `ast3`.** No `ast.py` there.
- **The held-out set is data, not code.** Each root has `heldout.json`: `{"version": "ast3", "freeze": "<commit>",
  "runs": ["xor/seed6", …]}`. The report's reproduction section for the newest version reads it. That replaces
  hard-coded seed rules such as "seed 2" and "seeds 3–5" for `ast3`; the older sections keep theirs.

## Left to the builder

- Module layout. The tanh rewrite fits next to the model code, since it is a pure `MLP → MLP` function.
- How `refactor` keeps "the net the routes see" apart from "the net the checker uses".
- The report layout, within Goal 7.
