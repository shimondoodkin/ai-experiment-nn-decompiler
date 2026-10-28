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

## Open questions and next steps

- **Step 5: compile to C and measure CPU speed against the net.** Also pruning. The programs already
  compile back to exact nets, so straight-line vs branch code can be benchmarked directly.
- **Step 6: bigger models:**
  - byte or keyword detectors (fixtures and ASCII constants);
  - Game of Life rules;
  - small RL policies;
  - one MLP block inside a small transformer, for decompile-fix-reinsert where relearning from cases
    is not enough.
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
