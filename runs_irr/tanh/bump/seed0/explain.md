# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 3.4% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 3.4%, at the box edge (x0 = -1.2, x1 = -3)
- by distance from (0, 0), rings 0.423 wide, mean: +0.4 +0.5 +0.1 -0.2 +0.1 +0.4 +0.2 +0.0 -0.0 -0.1; largest: +1.0 +1.1 +0.9 +1.4 +2.3 +2.4 +1.9 +3.4 +2.5 +2.1
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 +0.4 +0.2 -0.0 +0.2 +0.2 +0.1 +0.1 +0.0 +0.1 +0.1 +0.2; largest: +1.9 +1.6 +2.8 +2.3 +2.0 +2.4 +1.8 +1.9 +3.4 +2.2 +2.5 +2.1

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.025 * exp(-1.078 * x0 * x0 - 1.046 * x1 * x1 + 0.003583 * x0 * x1 - 0.003317 * x0 - 0.01026 * x1) + 0.002928
```

Clean:

```python
def net(x0, x1):
    return 1.025 * exp(-1.078 * x0 * x0 - 1.046 * x1 * x1 + 0.003583 * x0 * x1 - 0.003317 * x0 - 0.01026 * x1) + 0.002928
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00155, -0.00491), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.3%, inside the box (x0 = -0.0913, x1 = -0.0786)
  - by distance from (-0.00155, -0.00491), rings 0.424 wide, mean: +2.5 +0.7 -0.9 -1.0 -0.1 +0.6 +0.5 +0.3 +0.3 +0.2; largest: +3.3 +2.6 +2.3 +2.2 +2.2 +2.5 +2.2 +3.1 +2.3 +2.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.4 +0.2 -0.1 +0.2 +0.1 -0.0 +0.2 +0.2 +0.2 +0.1 +0.0; largest: +3.1 +3.1 +3.1 +3.1 +3.2 +3.2 +3.3 +3.3 +3.2 +3.2 +3.1 +3.1

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0205, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 76 | 0.513 | 1.5 | 0.33 | 0.16 | 0.19 | 0.16 | 0.15 | 0.744 | - | none (best 0.0883) |
| 1 | -124 | 0.543 | 1.48 | 0.31 | 0.19 | 0.22 | 0.15 | 0.14 | 0.776 | - | none (best 0.0923) |
| 2 | 168 | 1.26 | 2.37 | 0.31 | 0.25 | 0.30 | 0.13 | 0.00 | 0.598 | + | none (best 0.0674) |
| 3 | 100 | 1.24 | 2.24 | 0.33 | 0.24 | 0.28 | 0.15 | 0.00 | 0.539 | + | none (best 0.0652) |
| 4 | 36 | 0.332 | 2.91 | 0.08 | 0.27 | 0.43 | 0.20 | 0.03 | 0.419 | + | none (best 0.0503) |
| 5 | 178 | -0.49 | 1.49 | 0.17 | 0.16 | 0.19 | 0.16 | 0.33 | 0.692 | + | none (best 0.0826) |
| 6 | -143 | -0.537 | 1.52 | 0.13 | 0.14 | 0.23 | 0.19 | 0.30 | 0.792 | + | none (best 0.0943) |
| 7 | 136 | -1.1 | 2.63 | 0.01 | 0.12 | 0.35 | 0.32 | 0.20 | 0.568 | - | none (best 0.0683) |
| 8 | 96 | -0.45 | 1.64 | 0.15 | 0.17 | 0.21 | 0.17 | 0.30 | 0.885 | + | none (best 0.105) |
| 9 | -164 | 0.488 | 1.52 | 0.32 | 0.16 | 0.20 | 0.16 | 0.15 | 0.743 | - | none (best 0.0885) |
| 10 | 61 | -0.917 | 2.77 | 0.01 | 0.14 | 0.38 | 0.32 | 0.16 | 0.805 | - | none (best 0.0968) |
| 11 | 115 | 0.541 | 1.53 | 0.32 | 0.18 | 0.21 | 0.16 | 0.13 | 0.753 | - | none (best 0.0896) |
| 12 | -26 | -0.503 | 1.78 | 0.11 | 0.18 | 0.24 | 0.21 | 0.26 | 0.936 | + | none (best 0.112) |
| 13 | -46 | 0.551 | 1.35 | 0.33 | 0.18 | 0.21 | 0.13 | 0.15 | 0.593 | - | none (best 0.0706) |
| 14 | -152 | -1.28 | 2.13 | 0.01 | 0.12 | 0.28 | 0.25 | 0.34 | 0.356 | - | none (best 0.0427) |
| 15 | 159 | -0.436 | 1.53 | 0.15 | 0.17 | 0.21 | 0.17 | 0.30 | 0.302 | + | none (best 0.036) |

## Groups

- ring: 5 units (0, 1, 9, 11, 13) with gates at distance 0.49-0.55, switch width ~1.5, magnitude ~0.74 each, negative
- same line: units 4, 6 (gate angle 36 degrees, offset 0.332)
- the rest: units 2, 3, 5, 7, 8, 10, 12, 14, 15
