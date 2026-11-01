# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

No formula passes the check. Near formula (conic): it decides 0.0% of the checker's inputs differently where the net is confident.

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.2466 * x0 * x0 + x1 * x1 - 0.00505 * x0 * x1 - 0.2533 * x0 - 0.001297 * x1 < 0.9311
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.25% (the dirt), faithful and the net 0.39% (what no conic follows), clean and the net 0.40%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | -0.00505 | 4.76% |
| x0² | 0.25 | 0.2466 | 3.22% |
| threshold | 0.95 | 0.9311 | 1.99% |
| x0 | -0.25 | -0.2533 | 1.04% |
| x1 | 0 | -0.001297 | 0.41% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (1.03, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 84 | 0.905 | 0.475 | 0.57 | 0.05 | 0.06 | 0.05 | 0.27 | 13.4 | - | none (best 1.6) |
| 1 | -133 | 0.977 | 0.603 | 0.58 | 0.08 | 0.08 | 0.06 | 0.20 | 12.8 | - | none (best 1.52) |
| 2 | -175 | 160 | 110 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00551 | + | constant |
| 3 | -149 | 148 | 98.4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0073 | + | constant |
| 4 | -175 | 164 | 110 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00474 | + | constant |
| 5 | 7 | -192 | 111 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00169 | - | constant |
| 6 | 155 | -2.04 | 0.925 | 0.04 | 0.05 | 0.10 | 0.10 | 0.71 | 14.3 | + | none (best 1.7) |
| 7 | 3 | -159 | 109 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00569 | - | constant |
| 8 | 104 | -1.05 | 0.559 | 0.22 | 0.06 | 0.07 | 0.06 | 0.58 | 13.9 | + | none (best 1.65) |
| 9 | -98 | 0.797 | 0.487 | 0.55 | 0.05 | 0.06 | 0.05 | 0.28 | 13.3 | - | none (best 1.57) |
| 10 | 10 | -181 | 110 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00392 | - | constant |
| 11 | 52 | 1.53 | 0.689 | 0.68 | 0.08 | 0.08 | 0.06 | 0.11 | 12.6 | - | none (best 1.5) |
| 12 | -177 | 161 | 110 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00445 | + | constant |
| 13 | -26 | -1.12 | 0.696 | 0.17 | 0.07 | 0.10 | 0.08 | 0.58 | 12.4 | + | none (best 1.48) |
| 14 | 5 | -168 | 110 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00451 | - | constant |
| 15 | 108 | 0.806 | 0.482 | 0.56 | 0.05 | 0.06 | 0.05 | 0.27 | 13 | - | none (best 1.55) |

## Groups

- negligible: units 2, 3, 4, 5, 7, 10, 12, 14 (magnitude 0.0378 together; a quarter of the tolerance is 0.257)
- the rest: units 0, 1, 6, 8, 9, 11, 13, 15
