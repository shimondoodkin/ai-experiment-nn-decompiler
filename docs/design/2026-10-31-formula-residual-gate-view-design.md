# Near formulas, residuals and the gate view

Status: approved in chat ("Approve, write plan"), 2026-10-31. Plan: `docs/plans/2026-10-31-formula-residual-gate-view.md`.
Part of the irrational-functions project
(`2026-10-30-irrational-functions-design.md`); it changes that project's development step (Task 8 onward), before the
freeze.

## Why

The irrational development batch was stopped by the user after 18 of 108 nets (2026-10-30). The user's reason: "some of
the result are not good at all and the output is even not helpful". Where the formula passes, the output is good
(sigmoid circle 3/3, product 3/3: 18–25 nodes against ~466). Where it does not (2 of the 3 sigmoid dist nets, all 3 bump nets), `ast5.py`
is the net itself with a few units rounded, which tells a reader nothing.

What a closer look at `runs_irr/sigmoid/dist/seed0` found (2026-10-30/31, by hand, in chat):
- **The net is close to the formula, but not within 2%.** It is within ±0.06 of `r = sqrt(x0² + x1²)` from r = 0.5 to 3,
  is 0.33 at the origin (a rounded tip), and is slightly lopsided (centre shifted, x1 about 1% steeper). Every rounded
  radial formula tried is off by 0.10–0.14 (2.5–3.5% of the output range); the check allows 0.08.
- **Least squares is the wrong fit for a max-error check.** A minimax fit (Lawson reweighting, from the sigmoid agents'
  notes) brings `sqrt(quadratic) + d` from 2.8% to 1.9%, with unrounded coefficients.
- **The user's gate view explains the rest.** Each sigmoid unit read as a *gate* (the line where it switches, its
  distance and width), a *region* (which part of the S-curve the box uses: off, low knee, middle, high knee, on), a
  *magnitude* and a *direction* (sign). The 16 units fall into: 5 full steps on a ring at distance 0.7–0.9, 5 rounded
  ramps on a ring at 2.4–3.6, 4 small units that are a line or a constant, and 2 negative mostly-on corrections.
- **Single units do not simplify; groups do.** Only the 4 small units have a simple stand-in (constant, line, rounded
  hinge) within a quarter of the tolerance. The structure is in the groups (the rings).

## Goals

1. **Minimax fitting in the family route** (`nndc/heuristics/families.py`). After each family's least-squares fit, a
   minimax refinement by Lawson reweighting (weights multiplied by the absolute error and renormalised, then a weighted
   refit; 40 rounds). The family keeps whichever of the two fits has the smaller max error. Hinge refits get the same
   refinement. The checker still decides.
   - *My call:* refits inside snapping also use Lawson, with 10 rounds, on the snapping subsample; a least-squares refit
     after a snap would undo the minimax gain. If that makes the route slower than 10 minutes on a 2-D net, it falls
     back to least-squares refits during snapping plus one 40-round minimax refit of the free parameters before each
     program is emitted, and the change is recorded as an amendment.
2. **A near formula when no formula passes** (regression nets). For each family, the route also keeps the best program
   it could snap when the 2% check fails: snapping stops at the coarsest level whose max error stays within
   `max(tol, 1.25 × the family's minimax error)`, so rounding adds at most a quarter to the error. Correction terms are
   tried as now (up to 4). It is recorded in `ast5_trace.json` (`near`: family, program, max error as a share of the
   output range, terms) only when no family program passes. It is **not** a program version: it fails the check, so
   `results.json` does not list it and acceptance does not change.
   - *My call, choosing it:* candidates are every family's program at every term count it reached. Among those within
     3% of the output range, the fewest correction terms wins, then the smaller error, then fewer nodes; if none is
     within 3%, the smallest error wins. Extra hinges almost always lower the error, so ranking by error alone would
     favour 4-term programs of any family (on the pinned net, five families reach 2.1–2.5% with 4 terms).
   - *My call:* the 1-D `identity` family is never a near formula (it is a hinge program); the 2-D `quadratic` family is
     (it is `product`'s formula).
   - *My call, binary nets:* the conic route's program is the near formula when it fails the check; its error is the
     share of the checker's inputs decided differently outside the unsure zone. Near-formula counts in the report and
     prediction 2 cover regression nets only.
3. **The residual of the near formula**, described blind (from the net and the formula only):
   - its max error and where it is: at the box edge (within 5% of a face) or inside;
   - for 2-input nets, by distance from the formula's centre (the vertex of its inner quadratic, else the box centre)
     in 10 rings, and by angle in 12 sectors; a ripple shows as an angular pattern;
   - for 1-input nets, by position in 10 equal slices.
4. **The gate view** (`nndc/gate_view.py`), for one-hidden-layer nets (ReLU, sigmoid, tanh rewritten as sigmoid):
   - **per unit:** gate line (angle of w, offset −b/|w|, switch width 4/|w|; ReLU: the kink line), the share of the box
     in each zone (sigmoid: off z < −4, low knee −4..−1.5, middle −1.5..1.5, high knee 1.5..4, on > 4; ReLU: off,
     active), magnitude = the range of the unit's term (output weight × unit) over the box, sign of the output weight,
     and the simplest stand-in whose max error is within a quarter of the tolerance. The stand-ins, simplest first:
     constant, line, step, clamp (the existing ones in `nndc/heuristics/sigmoid.py`), then rounded hinge
     (*my call:* a fitted `k·relu(z − t) + c` in the unit's own z), else none;
   - **tolerance:** regression, 2% of the net's output range; binary, 2% of the range of the raw score
     (output − threshold) over the box (*my call*); magnitudes are in the same units;
   - **groups**, a partition, assigned in this order (*my call*; a unit goes to the first group it qualifies for):
     1. *negligible:* magnitude under a quarter of the tolerance (their summed magnitude is reported too);
     2. *lines:* the stand-in is a constant or a line (summed into one linear term);
     3. *rings:* units with the same output sign and positive offsets, sorted by offset and chained while each offset
        is within 15% of the previous one (the chain also ends when its largest offset exceeds 1.35× its smallest),
        magnitudes within a factor of 2 of the chain's median, at least 3 units, and no gap of 180° or more between
        neighbouring gate angles. Offsets are signed: a unit with a negative offset is mostly on, and is never in a
        ring;
     4. *same line:* gate angles within 10° (same or opposite direction) and offsets within 5% of the box's width, as
        kink merge pairs kinks;
     5. *the rest.*
   - computed on a uniform sample of the box (seed 0, 100 000 points).
5. **`explain.md` per run**, written by a new command `nndc explain RUNS… | --all --root R`:
   - the formula: the passing program if there is one, else the near formula with its error and residual summary;
   - the gate view: the unit table and the groups, one line per group in words (for example "inner ring: 5 steps at
     distance 0.7–0.9, magnitude ~1").
   It reads only the dump, the weights, the blind spec and `ast5_trace.json`, so it is as blind as the tool. It records
   `explain` in `run_git.txt`.
6. **Report** (`runs_irr/REPORT.md`): the results table gains a "near formula" column (family and error, for nets
   without a found formula), links each run's `explain.md`, and the "Snapping to formulas" table gains "near formula
   within 3%". The task-aware analysis adds `near_family_is_task_family`.
7. **Then the development step again:** the 108 built development nets stay (not retrained). `ast5` (with minimax and
   near formulas), `pysr_irr`, `irr-analyse` and `explain` run on all of them; the 18 earlier `ast5` results are
   replaced, since the route changed before any freeze. The rest of the irrational plan (freeze, held-out, write-up)
   follows unchanged.

## Pinned test case

`runs_irr/sigmoid/dist/seed0` (development net, already built). The gate view must find: an inner ring of 5 units
(offset 0.7–0.9), an outer ring of 4 or 5 units (offset 2.4–3.6), at least 2 negligible units, and the 2 negative
units (7, 14) in neither ring. Either a sqrt program passes the check, or the near formula is the sqrt family within
3%.

## Predictions (written before any code of this design)

- With minimax, the formula is found on at least 2 of the 3 sigmoid dist nets (it was 1 of 3).
- On at least 90% of the learned regression development nets, either the formula is found or the near formula is of
  the task's family and within 3%.
- On the 18 nets already run, `ast5` with minimax finds the formula on at least as many nets as before, and is never
  larger where both pass.

## Non-goals

- Writing hinge sums as if-chains, and nested conditions for larger nets (the user's ideas, 2026-10-30). They come
  next, in their own design, built on the gate view's groups and zones.
- Changing the 2% check, or accepting a near formula as a version.
- Two-hidden-layer nets, argmax nets and more than 2 inputs in the gate view.
- Turning a ring into a radial formula automatically (for now the ring is named; the near formula covers the radial
  part).

## Anti-goals

- Any change to `ast4`, the ReLU results or any accepted program of an older version.
- Task names, targets or families entering `explain.md` or the gate view (they stay blind; only `irr-analyse` and the
  report are task-aware).
- Tuning on held-out nets (none exist yet; the freeze comes after this).

## Constraints

Everything in CLAUDE.md, as in the irrational design. `explain` and the gate view are NumPy only.

## Left to the builder

The `explain.md` layout, the wording of group lines, how the Lawson loop guards against zero errors, and the module
split.

## Amendments

- **2026-10-31, faithful fit in `explain.md`** (the user: "show user faithful formula with dirt terms also ... let
  user see where error of the net comes from"). A new section, after Formula, for the family of the formula shown
  there (the passing family or conic program, else the near formula's family):
  - the family's 0-term minimax fit with every fitted value kept (4 significant digits), next to its clean
    (snapped) version;
  - the net minus the clean formula, split into **distortion** (clean minus faithful: the dirt) and **ripple**
    (faithful minus net: what no formula of the family follows), each with its largest error and the slice or
    ring/sector profile;
  - the dirt table: each value whose clean version differs, and its effect, the largest change over the box when
    only that value is changed (% of the output range; for a conic, % of the clean threshold). Changes can offset
    each other: on sigmoid/exp/seed0, scale 1.035 and inner slope 0.983 move the output 3.6% and 3.4% alone, 0.4%
    together;
  - conic nets: the share of the box where clean, faithful and the net decide differently, in pairs.

  It is report-only: `explain` re-runs the family route's own fit (`families.faithful_fit`, deterministic, the same
  fit `ast5` made). `ast5`'s programs and traces do not change, so no rerun is needed. This changes the constraint
  above: `explain` now uses SciPy through that fit; the gate view stays NumPy only. Neither runs in blind folders.
- **2026-10-31, first development numbers** (data commit `2962329`, `runs_irr/REPORT.md`). Predictions, as written:
  - formula found on at least 2 of the 3 sigmoid dist nets with minimax: not met (1/3);
  - formula found, or a near formula of the task's family within 3%, on at least 90% of the learned regression nets:
    not met (79/88, 89.8%);
  - on the 18 nets run before minimax, at least as many formulas and never larger where both pass: not met (10
    before, 10 now; larger on 1 of 18).
