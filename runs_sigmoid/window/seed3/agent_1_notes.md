# Agent notes

## What the network does

Output is binary (class 1 iff score > 0), one input x0 in [-2, 2]. Each hidden sigmoid has a
steep slope (|w| 10-14), so each is effectively a step at x = -b/w:

| unit | step at | direction | output weight |
|------|---------|-----------|---------------|
| h1_0 | -0.5034 | falls (1 for x < t) | -7.93 |
| h1_1 | -0.5041 | falls | -7.85 |
| h1_2 | +0.4925 | rises (1 for x > t) | -5.68 |
| h1_3 | +0.4964 | falls | +9.23 |

h1_0 and h1_1 are near-duplicate steps (same direction, thresholds 0.0007 apart) and merge into
one step of weight -15.78 at about -0.5. h1_2 and h1_3 sit at about +0.5 and both push the score down
for x > 0.5. The score is piecewise constant:

- x < -0.5: -15.78 + 9.23 - 1.76 = -8.3 (class 0)
- -0.5 < x < 0.5: 9.23 - 1.76 = +7.47 (class 1)
- x > 0.5: -5.68 - 1.76 = -7.44 (class 0)

The net's exact zero crossings (bisected) are x = -0.49496 and x = +0.49552. Both are within 0.005
of +-1/2, so I snapped them to one shared threshold, giving a symmetric band.

## Result

`agent.py`: `return 0.5 - abs(x0)`, so class 1 iff |x0| < 1/2.
`check.py` says PASS (snap): 0 disagreements where the net is confident. All disagreements are in
x0 in (-0.5, -0.495) and (0.4955, 0.5), inside the net's unsure zone.

I also checked an unsnapped variant with the raw thresholds, `-0.4949588 < x0 < 0.4955232`
(scratch/exact.py). It gets PASS (exact), but it only copies the net's training noise, so I kept the
snapped formula.

## Heuristics

- For a 1-input binary net, first scan the score on a dense grid and bisect its sign changes. The
  number and positions of the crossings tell you the whole decision rule before you read the units.
- A sigmoid with |w| times the input range >> 1 is a step at -b/w. Write each unit as (threshold,
  direction, output weight). Then the score is a sum of steps, and the plateau values show which
  units matter.
- Steps in the same direction whose thresholds are within about 1% of the range are duplicates: add
  their output weights and treat them as one step.
- A band of class 1 between two crossings at -t and +t (to within a few thousandths) is |x| < t.
  Write it with abs and one shared threshold. Returning a margin-like score (t - abs(x)) keeps the
  decision `raw > 0` readable.
- Snap thresholds to the nearest half-integer or simple fraction when the offset is well below the
  net's transition width (here about 0.005 against a sigmoid width of about 0.1). The checker's unsure
  zone takes up such offsets.
- An unsnapped program with the bisected crossings gives an exact pass for reference. It shows the
  snap only moves the boundary, not the topology.
