# Conclusions and notes, step by step

Hand-written, unlike `REPORT.md`, which `nndc report` regenerates. This file records what each step
taught us, what surprised us, what did not work, and the open questions. Most of it is not visible in
the result tables. Numbers come from the reports; observations about how agents worked come from their
notes and final reports. Those are first-hand but not systematically measured.

## Step 1: decompiler and checker

**What we learned**
- Writing a ReLU net as code is lossless and mechanical: every `dump.py` passes the checker exactly.
  The hard part is the checker, not the decompiler.
- The checker must compare with the net, never with the target. It must also say *where* a disagreement
  is allowed:
  - an exact match;
  - a snap tolerance for regression (2% of the output range);
  - for decisions, an "unsure zone" where the net's own score is small.

  Without the unsure zone, every snapped boundary fails on a sliver of points the net itself barely
  decides.
- Random inputs are not enough to test a program against a net. Points bisected onto the net's own
  activation boundaries find almost all real counterexamples. Uniform points rarely land near a kink.

**What did not work at first**
- **float32 ties:** binary nets whose decision line passes exactly through grid points gave float32 and
  float64 "disagreements" that were pure rounding. We fixed this with a tie tolerance, which became a
  propagated float32 rounding bound.
- **Exhaustive vs sampled domains:** the checker needs both. It enumerates small discrete domains
  (≤ 1e7 points); everything else is sampled and labelled "empirical", so a PASS is evidence, not proof.

## Step 2: Claude Code refactors the dump

**What we learned**
- A task-blind agent given only the dump, neutral input names and a checker recovers intended
  algorithms:
  - 29 of 29 models pass the check (12 exact), and 26 of 29 recover the true formula;
  - median of 3 went from a 2,500-node dump to the textbook `max(min(a, b), min(max(a, b), c))`;
  - tax/seed2 went to exact brackets 0.1 / 0.2 / 0.4 at 20 and 80.
- **The refactored program is often better than the net:** in 16 runs it was closer to the true task
  than the net it came from. Snapping removes the net's blur. Nobody told the agent the target.
- **When the agent "fails" it is usually the net's fault:**
  - tax/seed0–1: the net itself learned blurred brackets, and the faithful program copies them.
  - xor/seed2: the net never learned xor, so the honest program is not xor.

  The checker is right to reject the "intended" formula there.

**Surprises**
- Agents guess the domain from input ranges alone. Iris was recognised every time; tax was recognised
  as "income brackets".
- **The audit:** transcript audits for paths outside the blind folder produced many false positives:
  `/len` and `/print` in Python code, `/50` as division, `~` in prose. Only two real flags appeared
  (neareq seeds 0–1 wrote scratch files to `/tmp`). Path auditing needs human review, not a rule.

## Step 3: the deterministic AST tool

**What we learned**
- **Coverage:** most of what the agent does is mechanical.
  - **Results:** a tool built only from the heuristics the agents wrote down, frozen before the
    held-out run, reproduced Claude on 5 of 6 held-out nets and 21 of 21 in-sample ones.
  - **Moves it encodes:** pruning dead and always-on units, merging duplicates, snapping, ReLU
    motifs (abs, max, min, clamp, hat), 1-D spline reading, boundary measurement, truth tables, a
    small formula fit.
- **The gap is recognition, not mechanics:** only Claude got tax/seed2's 0.1 / 0.2 / 0.4. The tool
  kept a valid but blurred max of three lines. Seeing "this is a tax schedule" and snapping to it is
  the step that needs the LLM.
- **Selection rule:** fewest nodes first, exactness only as a tiebreak. "Exact first" made snapping
  unreachable, because the unsnapped program is always exact.

**Open**
- The spline route's intercept anchoring is weak. It was found on a held-out trace and deliberately
  not fixed, to keep the held-out result honest.

## Iris (dataset side run, steps 2–3)

**What we learned**
- Every Iris net, 4-feature or petal-only, reduces to an if-chain of two linear tests.
- **AST tool:** v1 of the decision-list route passed only 1 of 4 held-out nets. v2 adds two moves
  from the agents' notes: two half-planes joined by `or` for a kinked boundary, and thresholds centred
  in the range the confident points allow. v2 passed 6 of 6 fresh held-out nets.
- **Not the textbook rule:** the nets agree with the textbook petal rule on the 150 flowers but only
  on 58–70% of the input box. Off the data, the nets' boundaries are arbitrary extrapolation, so
  "faithful to the net" and "the botanist's rule" differ there.

**The fail-and-fix loop**
- The user noticed that agents first write a program from the general structure, then repair it from
  the checker's counterexamples.
- The checker logs show this loop only on Iris (7 of 12 runs got a FAIL), never on the formula nets.
- In the literature this is counterexample-guided synthesis with a sampling checker; see
  `reports/Counterexample guided program refinement.md`.

## Step 4: comparison and decompile-fix-reinsert

**Baselines** (29 formula nets, held-out seed 2 shown in `REPORT.md`)
- **Regions:** exact region decomposition is exact and enormous: a median of 71–141 nodes, and it
  timed out on median3. Exactness alone is not readability.
- **Tree:** decision-tree distillation passes the checker on 17 of 29 formula nets but on none of the
  12 Iris nets, even at 512 leaves. It recovers the formula only on xor. Axis-aligned cuts approximate
  slanted boundaries and smooth functions badly.
- **PySR:** it recovers 13 of 29 raw and 20 of 29 after this repo's snapping, so snapping is a
  missing ingredient of formula search. It fails on 4-feature Iris: long slanted decision lists are
  hard for a size-25 expression search.
- **Claude from samples vs from the dump:** Claude from input/output samples only did about as well
  as Claude reading the dump (25 vs 24 of 29 recovered).
  - On these small tasks, reading the weights is not what makes the agent succeed; 2,000 samples
    suffice.
  - Where it mattered: the dump gave tax/seed2's exact 0.1; samples gave 0.12.
  - It costs more feedback: on Iris the samples-only agents called the checker a median of 14 times,
    against 4 for the dump agents.

**How the agents worked without the dump** (from their notes)
- They rebuilt the structure themselves:
  - fitting a small ReLU net to the samples;
  - RANSAC over the samples to find the linear pieces;
  - fitting score differences only near each class boundary;
  - sorting a 1-D input and reading off slope changes;
  - grid-printing a class map before fitting anything.
- In effect they decompiled a surrogate net of their own. That is why the dump made little
  difference: the structure is easy to recover from samples at this size. We expect it to matter
  more as inputs grow.

**Decompile → fix → reinsert** (the user's idea)
- **Planted bugs:** all fixes were right on the held-out cases, and the rebuilt nets decide exactly
  like the fixed programs. The fix agent wrote the true function for absdiff, median3 and xor; for
  window the decompiled program was already right; for tax it was right except below zero (see below).
  - The median3 decompile agent found the planted bug by itself (median + 1.3 inside a box covering
    10% of the domain) and wrote it in a comment. The fix agent then used that comment.
  - The one imperfect fix, tax, wrote the first bracket as `relu(x0)`, which is wrong below zero,
    where no training data was.
- **Weight-space repair, given the same cases:** fine-tuning fixes about 75% of held-out failures and
  breaks a few percent of correct answers. The last-layer LP repair is often infeasible.
- **Iris:** the remaining errors are overlapping classes, not a bug in the net.
  - The petal-only agents called the duplicated flower with conflicting labels "label noise" and
    stopped at 97–98%. That is the right call.
  - The 4-feature agents reached 100% with a small zone rule around 3 flowers, and on one net that
    broke a held-out flower: overfitting, as the repair literature warns.
- **What this does not show yet:**
  - The agent fixed the planted bugs equally well *without* the decompiled program (cases only).
    These tasks are simple enough to solve from labelled data, so the experiment cannot yet say that
    decompiling helps the fix.
  - That needs a block whose function cannot be relearned from a few thousand cases, which is the
    step-6 setting.
- **Compile-back is the quiet success:** every program the agents wrote compiles back to a ReLU net
  exactly (relu/min/max/abs and decision lists need no approximation; only jumps need a ramp). So
  "put it back" is solved for this program language.

**Process notes**
- 74 blind agent runs went through cleanly with one fresh subagent per folder. Several harness bugs
  were found only by looking at real collected data:
  - the collect printout;
  - the audit flagging every agent's own "I did not open oracle.bin";
  - an LP repair undone by float32 rounding;
  - stale data after a metric change.

  Always read a sample of the raw outputs, not just the summary lines.
- **Decisions made during step 4 that change how to read the results:**
  - **Tax seeds 0–1 had nothing to fix:** with no failing cases at the 2% tolerance, they got no fix
    agents, so the natural-bug group rests on xor/seed2 alone.
  - **Two planted-bug nets never learned their bug** (tax and window, under 5%). They stayed in and
    are marked in the report, because their failures are ordinary training error.
  - **Audit exclusions:**
    - Both Claude rows keep runs with outside-path flags, which earlier reports reviewed by hand as
      mostly false positives.
    - A run is excluded only for touching a forbidden file. One samples-only Iris run did `ls _checker`.
  - **The special-case flag:** it fires only on repair sets of 20+ points, and it scores each
    comparison alone. So it misses conjunction zones like the Iris ones.
  - **Reinsertion success:** it means the rebuilt net decides like the program on every task case. The
    checker cannot judge it, because a jump compiles to a ramp of width 1e-6 of the margin, and the
    checker's boundary probes land inside it. Box mismatch, the share of uniform inputs inside ramps,
    is at most 0.005%.
  - **PySR on argmax nets** fits the logits and takes the argmax; the design predated Iris.
  - **LP repair:** it is verified after float32 rounding, and solved at 0.999 of the tolerance.

## Sigmoid nets and kink merge (side track, 2026-10-28 to 29)

Three projects, merged together:
- **A:** activation types in the code;
- **B:** 30 sigmoid nets with agents and sigmoid moves for the AST tool (`runs_sigmoid/`);
- **C:** kink merge, a new AST tool version `ast2`, with 15 fresh held-out nets.

Results are in `runs_sigmoid/REPORT.md` and `runs_sigmoid/iris/REPORT.md`.

**What we learned**
- **The code handles both activations in one path.** ReLU, sigmoid and tanh are a per-layer setting.
  - ReLU-only routes refuse other nets, and sigmoid moves run only on nets that have sigmoid units.
  - ReLU results stayed byte-identical through all three projects (equivalence script: 0 of 248 files differ).
- **Sharp sigmoid nets are as easy as ReLU nets.** On xor and window the units are steep steps, and both the agents
  and the tool recover the formula. Neareq too, for the agents.
- **Smooth nets bend the formula in three ways.** Measured against the true formula, as a share of the output
  range:
  - a ripple everywhere, about 1% mean (ReLU nets: 0.02%), because straight lines are built from S-curves;
  - rounded kinks, 2–3 times the mean error near the bend of `|x0−x1|`;
  - the largest: flattening at the edge of the input box, where the output is extreme. Absdiff at (3, −3) gives
    5.4 instead of 6 (9–14%), and tax flattens above income ~185.

  A sum of sigmoids cannot keep climbing to the edge; a ReLU can. So on absdiff, median3 and tax the formula
  fails against the net, and every faithful program is long.
- **The edge error is one term (the user's idea, project C):**
  - each sigmoid becomes 17 straight pieces, so an 8-unit net is 128 ReLU kinks;
  - kinks on the same line add up, and about half of the lines nearly cancel;
  - what is left is the formula's kinks, each split into a pair around the true place, plus one negative kink near
    the edge;
  - on absdiff: kinks at `x0−x1 ≈ ±0.25` and at `±5`. On tax: a bracket near 75 and an edge kink near 185–196.

  The same structure showed up on the 15 held-out nets. Blind agents found it on their own: tax agents added "a cap
  at 60" or "a knot at 190" to the bracket formula.
- **Faithful vs snapped (the open policy question):** the net is the formula plus one edge correction. Dropping
  that term gives the formula, which fails the checker against the net. Keeping it gives a short faithful
  program.
  - Proposal (to confirm with the user): acceptance stays faithful, as now.
  - The report shows the faithful program next to `formula.py`, so the difference is one named term rather than
    a long dump.
- **Where kink merge wins** (route won, development seeds 0–2 / held-out seeds 3–5):
  - neareq 3/3 and 3/3;
  - absdiff 2/3 and 1/3;
  - tax 1/3 and 0/3. Knot-fit is smaller there, but kink merge's own program passed on all 3 held-out tax nets.

  Before kink merge, neareq and absdiff fell back to the dump (230–240 nodes). Now they are 66–116 nodes.
- **Reproducing the agents.**
  - The project B tool, frozen at `15b943b`, on its held-out seed 2 reproduced Claude's best valid attempt on 2 of
    3 runs. Counting audit-flagged attempts too: 3 of 6 held-out and 5 of 12 in-sample. Its `ast.py` files were
    removed later, see below; this count is from the project B report in git history.
  - `ast2` reproduces Claude's best valid attempt on 8 of 12 held-out nets and 5 of 7 development nets
    (`runs_sigmoid/REPORT.md`).
- **Agents still beat the tool on readability.** They write `0.2 - abs(x1 - x0)` for neareq, while kink merge writes
  three weighted kinks with unrounded coefficients that pass the same check.

**What did not work, or surprised us**
- **The spike's fixed line positions were not enough.** Refitting the strongest lines needed 6 lines to pass. The
  route also had to move line positions (an offset search), and that was slow until the search used 2,000 points.
- **Committed `results.json` files can be stale.** Re-evaluating changed `n_inputs` (+8 to +20) in several runs
  without changing any verdict. Re-run `evaluate_run` before every data commit, as the practical notes say.
- **Parallel agents shared one scratchpad** in project B, and the audit flagged it. A `scratch/` folder in each
  blind folder fixed this: 12 of 15 held-out audits were clean, and the other 3 were false positives.
- **`--name` could overwrite run files** until the final review. Now kink merge runs only for `ast2`, and only
  `ast`/`ast2` are accepted.
- **Superseded results:** the user decided that runs are results of code and git keeps them. So `ast2` replaced
  `ast` in every sigmoid run, and the project B `ast.py` files live in git history.

**Not done yet** (at the time; all three are handled in the next section: the review changes and tanh in `ast3`,
median3 as a spike finding)
- Two-layer sigmoid nets (median3): after expansion, their kinks bend, so they do not lie on lines.
- Tool changes the final review found. They need a new freeze and a fresh held-out set:
  - near-axis lines can split in two;
  - the unrounded candidate keeps lines that left the box;
  - the position tolerance should scale with the line, not the largest input range;
  - constants are written at full precision.
- Tanh, which is a scaled sigmoid and should reuse everything here.

## Tanh nets and `ast3` (2026-10-29)

These are the open items from the section above. Design: `docs/design/2026-10-29-ast3-tanh-design.md`. Results:
`runs_tanh/REPORT.md`, `runs_tanh/iris/REPORT.md` and the "Kink merge (ast3)" section of `runs_sigmoid/REPORT.md`.

**What we learned**
- **Tanh is a sigmoid in disguise:** `tanh(z) = 2·sigmoid(2z) − 1`.
  - Rewriting tanh layers as sigmoid layers (exact maths) let every sigmoid move run on tanh nets unchanged.
  - Checking against the original tanh net kept the result honest.
  - `ast3` passed on all 30 tanh nets. Kink merge won 7 of 10 development and 3 of 5 held-out tanh formula nets
    where it applies.
- **Tanh nets behave like sigmoid nets.**
  - Sharp tasks (xor, window, neareq) are recovered by the agents.
  - Smooth tasks (absdiff, tax, median3) get faithful programs: the formula plus correction terms.
- **"Formula plus corrections" is how agents write a smooth net, even with two layers.**
  - **Median3:** all three tanh agents wrote the median plus named terms: rounding at ties, flattening at the box
    corners, and an offset or a pull toward the mean. That is what kink merge does for one-layer nets, and kink merge
    cannot do it for median3 (see the spike below).
  - **Tax:** agents wrote the brackets plus an edge cap or knot.
- **4-feature Iris is harder with tanh.**
  - With sigmoid, a linear route passed on 4 of 6 nets: seeds 2–5, at 72–74 nodes. Seeds 0–1 fell back to the dump.
    With tanh no linear route passes on any of the 6, and the tool gives 434–484 nodes.
  - The agents found the reason: the boundaries bend near the box edges. They used two planes per class or a squared
    term, and wrote 93–310 nodes.
- **The review fixes (kink merge v2) changed little overall, but one prediction was wrong.**
  - On 45 sigmoid runs that have both versions, `ast3` was the same size as `ast2` on 37, smaller on 3 (absdiff seed
    4: 216 → 158 nodes) and larger on 5 (up to 81 → 118).
  - By the rule fixed before the results, `ast3` is worse on those 5 runs, so the `ast2` files stay next to `ast3`.
  - **Two of the four fixes cause the regressions, not one.** The final review re-ran the 5 runs with fix 3 (the
    position window) and fix 4 (the rounded constant) undone, one at a time and together:

    | run | `ast3` | constant unrounded | old position window | both undone (= `ast2`) |
    |---|---|---|---|---|
    | neareq/seed5 | 118 | 118 | 66 | 81 |
    | neareq/seed4 | 76 | 76 | 66 | 66 |
    | absdiff/seed0 | 86 | 86 | 86 | 80 |
    | absdiff/seed1 | 98 | 95 | 98 | 95 |
    | absdiff/seed5 | 89 | 89 | 89 | 86 |

    - The constant alone explains absdiff seed 1, and the window alone neareq seeds 4–5. The other two need both.
    - **Refuted prediction:** the design expected fix 3 to matter only for near-axis normals or mixed input ranges.
      But every kink line on absdiff and neareq is diagonal (`x0 − x1`), and for a diagonal line the new window
      `Σ|n_i|·span_i` is twice the old `max span` window. So it changes exactly the tasks where kink merge wins.
    - Fix 4's constant window `tol·R` can also take the whole 2% error budget at the coarse levels.
  - **Next version:** at each level, try both position windows and both constant treatments, and keep the smallest
    passing program. Cap the constant window well below 2%, and make the direction snap span-aware.
- **Held-out:** the frozen `ast3` passed on all 27 held-out nets (sigmoid seeds 6–8; tanh formula seed 2 and Iris
  seeds 3–5). The sigmoid route winners matched the prediction: neareq kink merge, tax knot-fit, xor and window as
  before.

**The median3 spike** (scratchpad, before the design)
- Expanding both sigmoid layers of median3 into 17-piece chords stays within 1.6% of the net.
- But the first layer's 256 kinks still form 147–224 lines in 9–14 directions. Nothing cancels at the first layer: the
  cancellation happens only after the second layer mixes the units.
- So line merging does not reach two-layer nets. The agents' "median plus corrections" programs suggest a different
  route: guess an order-statistic form, then fit corrections in the sorted values.

**What did not work, or surprised us**
- **A blind agent tried to kill every Python process** on the machine with `taskkill`, to stop its own background
  job. The permission rules blocked it, and it used its task tools instead. If it had gone through, it would have
  killed the tool runs.
- **Several agents guessed "Iris"** from the input ranges alone. Blindness held (no task information in the
  folders), but input ranges are themselves a hint.
- **The shell hook (`rtk`) rewrites `grep`**, which silently broke a collect loop. Use `awk` for filters in scripts.
- **Some trained nets did not learn,** with tanh more than sigmoid: tanh xor seed 1, window seeds 1–2, absdiff seeds
  0 and 2, Iris-petal seeds 2 and 4; sigmoid absdiff seed 7. They stay in and are marked. Agents found what such nets
  really compute ("always class 0", "just `x0`").

## `ast4`: the regression fixed (2026-10-30)

Design: `docs/design/2026-10-30-ast4-design.md`. Spike: `experiments/spikes/ast4_snap_variants/`.

**What we learned**
- **The `ast3` regression came from the search order, not the rounding itself.** Kink merge stopped at the first
  program that passed: fewest lines first, then coarse rounding first. `ast3` rounds more, so at the line count where
  `ast2` passed, its candidates failed, and it went on to finer levels or more lines.
- **Passing is not monotone in the rounding level.** One candidate passes at 5% rounding and fails at 2%. So "first
  pass, coarsest first" was partly luck, for `ast2` too.
- **The fix: search the whole level, keep the smallest.** At each line count, `ast4` checks every merge offset, snap
  level and snap variant. The variants are `ast3`'s or `ast2`'s position window, and a rounded or unrounded constant.
  It keeps the passing program with the fewest nodes.
  - In the spike this equalled the smallest passing program anywhere in the grid, on all 30 nets.
  - It costs up to 40 checks per line count.
- **Results:**

  | | runs | `ast4` smaller | larger / worse |
  |---|---|---|---|
  | against `ast2`, sigmoid development | 45 | 5 | 0 |
  | against `ast3`, sigmoid development | 60 | 9 | 0 |
  | against `ast3`, tanh development | 30 | 4 | 0 |
  | against `ast3`, fresh held-out (sigmoid seeds 9–11, tanh seed 3) | 20 | 4 | 0 |

  - Every prediction in the design held. The frozen `ast4` (`5ed6ee8`) passed on all 20 held-out nets.
  - The biggest gains are on neareq: 116 → 64 (seed 1) and 118 → 66 (seed 5). On held-out, 116 → 83 (seed 9) and
    tanh 81 → 67.
- **`ast4` is never worse than either older version,** so the `ast2` and `ast3` files can go (CLAUDE.md:
  superseded results may go). That needs the user's agreement and a commit of its own.

**Deferred** (from the `ast3` review, still open): route-internal checks use the rewritten tanh net; the direction
snap is not span-aware; `refactor` accepts any version string; `_heldout_rule` falls back to the old seeds-3–5 rule
for versions without `heldout.json` (now `ast3`, whose set is in `heldout_ast3.json`). From the `ast4` review:
- on an exact node tie, the route's dedupe keeps a duplicate source's first label, not its best one. That only
  changes which of two equally short programs wins, and the trace's label;
- `test_best_candidate_order` checks each tie-break alone, not their precedence.

## Open questions and next steps

- **Step 5: compile to C and measure CPU speed against the net.** Also pruning. The programs already
  compile back to exact nets, so straight-line vs branch code can be benchmarked directly.
- **Step 6: bigger models:**
  - byte or keyword detectors (fixtures and ASCII constants);
  - Game of Life rules;
  - small RL policies;
  - one MLP block inside a small transformer, for decompile-fix-reinsert where relearning from cases
    is not enough.
- **Smooth functions (the user's next idea):** can the decompiler and tools recognise functions that are not
  piecewise linear, such as `exp`, `sqrt`, or a circle `x² + y² < r²`? Next to research.
- **A harder fix population:** more planted-bug nets that learn their bug, and natural bugs with real
  failures, so that "with program" vs "cases only" can separate.
- **Deferred minors** (none affects current results):
  - `compile_back` flattening is exponential in long chains of non-returning `if`s, and it rejects
    `pass`;
  - the LP repair labels every failure "infeasible" and has no time cap;
  - the "more than a third flagged" rule is applied per table rather than over all 41 input/output
    runs;
  - PySR binary programs assume threshold 0;
  - two batch commands miss a few exception types;
  - a failed fix collect can leave one stray file.
