# Kink merge for sigmoid nets

Status: approved, 2026-10-29. Follows project B (`2026-10-28-sigmoid-nets-design.md`, branch `sigmoid`).

## Why

The project B tool cannot shorten the sigmoid nets that round the formula's corners: absdiff, neareq and
median3 stay near dump size. Tax is shortened by `knot-fit`, which only works for 1-input nets. The agents
stay faithful there, because the formula fails against the net.

The user's idea, after the corner analysis: replace each sigmoid by more straight pieces and see if terms
cancel. A throwaway spike tried this on absdiff and tax, seeds 0–2. The scripts are in the session scratchpad
(`pieces.py`, `pieces2.py`).

1. **Expand.** Each sigmoid became 17 straight pieces over the range its pre-activation takes on the input
   box. Every piece boundary is a ReLU kink, so an 8-unit net became 128 kinks. The result stays close to
   the net: max error 1.8% (absdiff) and 0.4% (tax) of the output range.
2. **Merge.** Kinks that lie on the same line add up:
   - on absdiff, every line was parallel to `x0 = x1`, and the 128 kinks became 16–64 lines;
   - on tax, they became 8–16 points;
   - about half of the lines nearly cancel: one unit's kink goes up, another unit's goes down.
3. **What is left** is the same on every seed:
   - the formula's kinks, each split into a pair around the true place (a rounded corner);
   - one negative kink near the edge of the box. That is where the sigmoids flatten, and it is the
     corner error found earlier.
4. **Refit.** A few kinks, refitted to the net, give short faithful programs:
   - **tax:** `0.1·x + 0.1·max(x−20,0) + 0.2·max(x−80,0) − 0.3·max(x−193,0)` is 0.8–1.0% from the net on
     all 3 seeds;
   - **absdiff:** `max(|x0−x1|, 0.3) − 0.55·max(|x0−x1| − 5, 0)` is 1.8% and 1.9% from the net on seeds 0–1,
     and 4.0% on seed 2.

   These shapes were picked by hand, knowing the formula. Refitting the strongest lines with their spike
   positions left fixed got to 1.9% only with 6 lines. So the tool must also move line positions, not
   only refit coefficients. Errors were measured on 100k samples, not with the checker.

**The user's direction:** write this up as a design for a new AST tool move.

## Goals

1. **A new AST route, `kink-merge`,** for nets with one hidden layer of sigmoid units and a regression or
   binary output. The steps, all with parameters fixed in this document:
   1. **Unit box.** Work in coordinates `u = (x − lo) / (hi − lo)`, so that tolerances mean the same for
      every input range.
   2. **Expand.** Each unit's sigmoid becomes `K = 17` straight pieces. The knots are uniform in `z` over
      `[z_min, z_max]`, the range of the unit's pre-activation on the box corners. This turns each unit into
      an affine term plus `K − 1` kinks of the form `strength · max(n·u + d, 0)`, with `|n| = 1`.
   3. **Canonical direction.** A kink whose normal points the "wrong" way (first nonzero component negative)
      is flipped, using `max(y,0) = max(−y,0) + y`; the `+ y` goes into the affine part. A kink whose line
      does not cross the box is affine there (always on) or zero (always off), and is folded in.
   4. **Merge.** Kinks with normals whose cosine is above 0.995, and offsets within `MERGE_OFF`, become one
      line. Its strength is the sum of the strengths, and its position is the strength-weighted mean.
      `MERGE_OFF` takes two values, 0.02 and 0.1, giving two candidate line sets.
   5. **Pick.** For `m = 1..8`, keep the `m` lines with the largest absolute strength.
   6. **Refine.**
      - **Offsets:** a coordinate search. Each kept line's offset is tried over ±0.1 in 21 steps, for 3
        rounds. Normals stay as merged.
      - **Coefficients:** for each tried set of offsets, the constant, the linear term and the line
        strengths come from the minimax LP that `knot_fit` already uses (`_minimax`). It is fitted to the
        net's output on 20,000 uniform samples from the box (fixed seed) plus the box corners.
      - **Binary nets:** the fit target is the net's raw score, and acceptance is the checker's, on decisions.
   7. **Snap,** in real input coordinates, once per tolerance `tol` in `SNAP_TOLS` (0.05, 0.02, 0.01,
      0.005), coarsest first, then the unsnapped fit. Each level is its own candidate. A unit-box snap would
      move tax's edge kink at income 193 (u = 0.965) to 200, outside the box, so the snap happens in real
      coordinates, with tolerances scaled to the input range.
      1. **Direction:** scale each line's real-coordinate normal so its largest component is 1, then
         `nice(c, tol)` the other components. They are ratios in [−1, 1], so an absolute tolerance is right.
      2. **Position:** write the line as `n·x = t` and replace `t` by
         `nicest_in(t − tol·span, t + tol·span)` (from `decision_list`), where `span` is the largest input
         range. For tax at tol 0.005 that is the simplest number in (192, 194).
      3. **Refit:** re-run the minimax LP for all coefficients on the snapped lines.
      4. **Coefficients:** replace each line strength and linear coefficient `s` by
         `nicest_in(s − tol·|s|, s + tol·|s|)`, a relative tolerance. Then refit only the constant, as the
         midpoint of the remaining error's range, which is its minimax value.

      A line whose snapped position leaves the box is folded into the affine part, as in step 3.
   8. **Write.** The program is `c + a·x + Σ s·max(n·x + d, 0)`, written back in the real input coordinates
      of the blind spec. Then the existing ReLU motif rewriting turns pairs on one line into `abs(...)` or
      `max(...)` where it applies.
   9. **Keep.** A candidate is kept only if it passes the checker. Within the route, the smallest passing
      `m` wins; across routes, the existing rule applies (fewest nodes, exactness breaking ties).
2. **The trace shows the cancellation.** `ast2_trace.json` records, per merge tolerance, how many kinks,
   lines and nearly cancelled lines there were. A line counts as nearly cancelled when it was built from at
   least 2 kinks and its net strength is below a quarter of the sum of their absolute strengths. The trace
   also records the kept lines (normal, offset, strength) of the winning candidate. This answers the user's
   question: which terms cancel, and which stay.
3. **A new tool version, without touching old results.**
   - The tool with the new route writes `ast2.py`, `ast2_trace.json` and a `run_git.txt` line `ast2:`.
   - `ast.py` files are never rewritten.
   - `evaluate_run` checks `ast2` as a version.
4. **Development and freeze.**
   - **Development set:** the existing sigmoid runs, seeds 0–2, of the one-hidden-layer formula tasks: xor,
     window, neareq, absdiff, tax.
   - Seed 2 was the project B held-out set. For this move it becomes development: the spike looked at
     absdiff and tax seed 2. The project B held-out result stays as it is, since its `ast.py` is not
     rewritten.
   - The parameters above are fixed now. Development may fix bugs, judged on the development set only. It
     adds no moves and changes no parameters after seeing results. Then comes the freeze commit.
5. **A fresh held-out set,** built after the freeze. Seeds 3, 4 and 5 of xor, window, neareq, absdiff and tax,
   with sigmoid hidden units, in `runs_sigmoid/<task>/seed3..5`. That is 15 new nets, with the same sizes and
   training settings as seeds 0–2. Each gets:
   - `formula.py`, as in project B;
   - one blind agent pass (15 subagents, one wave), with heuristics going to
     `runs_sigmoid/HEURISTICS_heldout.md`;
   - the frozen tool, which writes `ast2.py`.
6. **Report.** `runs_sigmoid/REPORT.md` (regenerated) gets a section "Kink merge (ast2)". It has one row per
   development and held-out net, with:
   - the winning route, its verdict against the net, and its nodes;
   - whether it recovers the formula;
   - the counts kinks → lines → nearly cancelled → kept;
   - the kept lines, in real coordinates.

   It also has the reproduction count for `ast2`, development vs held-out, and puts the `ast2` program next
   to `formula.py` for each formula net, so a reader can see which terms differ.
7. **Afterwards, not part of the build:**
   - the user and Claude read the result and add to the sigmoid section of `CONCLUSIONS.md`;
   - **retiring the old version.** The user's view: once `ast2` works, the old `ast.py` is not needed, and
     is kept only for the comparison. Removing it is a separate later step. It needs the CLAUDE.md
     "older runs stay" rule changed first, and the report must then stop reading `ast` for sigmoid runs.
     Until then, both are kept.

## Predictions (written before any tool code)

From the spike, I expect:
- `kink-merge` wins on absdiff and tax, development seeds 0–1;
- on held-out absdiff and tax, it passes on at least 4 of 6 nets;
- the kept lines on absdiff are the split kink near `x0 = x1` and an edge kink near `|x0−x1| ≈ 5`, and on
  tax the brackets near 20 and 80 plus an edge kink near 190;
- xor and window keep their project B winners, which are already small;
- neareq is unknown: it is binary and its boundary is curved.

## Non-goals

- **Nets with two or more sigmoid layers (median3).** After expansion, the second layer's kinks bend, so they
  no longer lie on lines. Argmax nets (Iris) are also out; they have `linear-scores`.
- **Snapping to the formula.** Dropping the edge kink would give the formula, but the program must pass
  against the net, as always. The report shows the difference; it does not change the policy.
- Tanh, changes to checker tolerances, `compile_back` for these programs, and step-4 baselines on the new
  nets.

## Anti-goals

- **Any change to an existing result.**
  - ReLU runs: the route runs only for nets with sigmoid units, and the equivalence script must report 0
    differences.
  - Sigmoid runs: no `ast.py`, agent file or checks log is rewritten.
- **Tuning after seeing held-out nets,** or building them before the freeze.
- **The tool seeing the task:** it reads only the dump, the blind spec and the weights, as now.
- **Deleting or overwriting older runs,** per CLAUDE.md.

## Constraints

- Everything in CLAUDE.md: experimental discipline, `data:` commits separate from code commits, older runs
  stay, and destructive commands only on scratch copies.
- **Agent scratch files.** Each blind agent gets a scratch directory inside its own blind folder. This is the
  project B lesson; `INSTRUCTIONS.md` stays the same otherwise.
- **The route's runtime** must stay reasonable inside `nndc ast`. 8 values of `m`, 2 merge tolerances, 3
  rounds of offset search and one LP per try add up to a few thousand small LPs per net. The builder may
  lower the LP size (fewer fit points) if this is slow, as long as it is fixed before the freeze.

## Decisions (my calls; the user can override)

- **`ast2` as a new version, not a rerun of `ast`.** The "older runs stay" rule forbids rewriting `ast.py`.
  The comparison with project B is also clearer this way.
- **`ast2` has its own counterexample file, `counterexamples_ast2.json`,** reused whenever `ast2` is
  re-evaluated. `counterexamples.json` stays the set for the older versions. If `ast2`'s points were added
  to it, re-evaluating would check the old `ast` and agent versions on more points, and an old verdict could
  change. (The user's suggestion: separate counterexample sets per tool version.)
- **Held-out seeds 3–5, 15 nets,** for the 5 one-layer formula tasks only. Median3 and Iris get no new seeds,
  because the route does not apply to them.
- **Agents on the new nets too,** so the tool can be compared with agents on unseen nets, as in step 3.
- **Snap in real coordinates, geometry first, then refit, then coefficients** (Goal 1.7). The reviewer
  pointed out that the same absolute tolerances mean very different things in unit-box and real
  coordinates. The scaled tolerances keep one meaning across tasks.
- **Branch:** `kink-merge`, cut from `sigmoid` (the user asked for a new branch).
- **Parameters from the spike:** 17 pieces and the 0.995 cosine are the spike's values. Merge offsets 0.02
  and 0.1 are the spike's two values. `m ≤ 8`, the ±0.1 offset search and the quarter rule for "nearly
  cancelled" are new; I picked them from the spike's numbers.

## Left to the builder

- Module layout (a new `nndc/heuristics/kinks.py` is the obvious place), names, and the CLI flag that makes
  `nndc ast` write `ast2`.
- How the minimax LP is shared with `knot_fit`.
- The report section's layout, within the content in Goal 6.
- The order of moves inside the route, as long as the parameters and the acceptance rule stay as written.
