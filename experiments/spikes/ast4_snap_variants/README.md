# Spike: kink-merge snap variants (2026-10-30)

Why `ast3` was larger than `ast2` on 5 of 45 sigmoid runs, and what fixes it. It led to `ast4`
(`docs/design/2026-10-30-ast4-design.md`).

- `spike.py RUN...` builds every kink-merge candidate of each run and checks it. It covers m = 1-8, both merge offsets,
  the 5 snap levels, and 4 snap variants: the position window (`ast3`'s line range or `ast2`'s max span) and the
  constant (rounded or not). It writes `out/<run>.json`; the folder is git-ignored and takes about 4.5 min per net.
- `analyse.py` compares route policies on those grids. The tool's size is the smaller of the policy's program and
  the best other route in the run's `ast3_trace.json`.

It ran on the 30 one-layer sigmoid formula nets of seeds 0-5 (xor, window, neareq, absdiff, tax), at commit
`08cddeb`. Its `ast3` policy gave the committed `ast3` sizes on all 30. Against `ast2`:

| policy | smaller | same | larger |
|---|---|---|---|
| `ast3` as committed | 3 | 22 | 5 |
| old window and unrounded constant (only fixes 1-2) | 1 | 29 | 0 |
| 4 variants per level, smallest | 4 | 26 | 0 |
| smallest at the first line count that passes (`ast4`) | 5 | 25 | 0 |

The last policy equalled the smallest passing program anywhere in the grid on all 30 nets.
