# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 5.9% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 5.9%, at the box edge (x0 = 0.2, x1 = -3)
- by distance from (0, 0), rings 0.423 wide, mean: +1.5 -0.2 -1.2 -0.9 -0.2 -0.0 -0.3 -0.4 -0.5 +0.4; largest: +3.0 +2.4 +3.2 +3.4 +3.0 +2.6 +5.2 +5.9 +4.8 +3.3
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.6 -0.7 -0.4 -0.0 +0.3 +0.4 +0.1 -0.3 -0.7 -1.3 -0.6 -0.6; largest: +5.1 +4.1 +4.4 +4.1 +3.3 +2.7 +2.4 +3.6 +5.6 +5.9 +2.8 +5.1

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.9789 * exp(-0.9706 * x0 * x0 - 0.9809 * x1 * x1 + 0.002348 * x0 * x1 + 0.01871 * x0 - 0.01176 * x1) + 0.004295
```

Clean:

```python
def net(x0, x1):
    return 0.9789 * exp(-0.9706 * x0 * x0 - 0.9809 * x1 * x1 + 0.002348 * x0 * x1 + 0.01871 * x0 - 0.01176 * x1) + 0.004295
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.00963, -0.00598), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 5.5%, at the box edge (x0 = 3, x1 = -0.91)
  - by distance from (0.00963, -0.00598), rings 0.424 wide, mean: +0.3 -0.4 -0.6 -0.1 +0.4 +0.4 +0.1 +0.0 -0.0 +0.9; largest: +2.2 +2.2 +2.2 +2.6 +3.1 +3.1 +5.0 +5.5 +4.1 +3.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 -0.2 -0.0 +0.2 +0.6 +0.7 +0.5 +0.1 -0.2 -0.7 -0.1 +0.1; largest: +4.7 +3.8 +4.8 +4.6 +3.8 +3.1 +2.8 +4.0 +5.2 +5.4 +2.0 +5.5

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0206, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -39 | -1.57 | 1.67 | 0.03 | 0.09 | 0.19 | 0.22 | 0.48 | 0.624 | - | none (best 0.0745) |
| 1 | 85 | 1.82 | 1.75 | 0.51 | 0.18 | 0.22 | 0.08 | 0.00 | 0.537 | + | none (best 0.0503) |
| 2 | 164 | 0.524 | 1.26 | 0.37 | 0.14 | 0.16 | 0.13 | 0.19 | 0.65 | - | none (best 0.0771) |
| 3 | 114 | 0.627 | 1.37 | 0.37 | 0.15 | 0.19 | 0.15 | 0.14 | 0.597 | - | none (best 0.0709) |
| 4 | -19 | -1.42 | 2.22 | 0.00 | 0.11 | 0.28 | 0.25 | 0.36 | 0.805 | + | none (best 0.0972) |
| 5 | -33 | 1.58 | 2.65 | 0.29 | 0.33 | 0.31 | 0.08 | 0.00 | 0.684 | + | none (best 0.0781) |
| 6 | -79 | -1.1 | 3.43 | 0.00 | 0.09 | 0.44 | 0.37 | 0.10 | 0.447 | + | none (best 0.0327) |
| 7 | -139 | 1.43 | 5 | 0.01 | 0.39 | 0.57 | 0.02 | 0.00 | 0.443 | + | none (best 0.0261) |
| 8 | 8 | -1.6 | 1.86 | 0.00 | 0.11 | 0.23 | 0.20 | 0.45 | 0.517 | - | none (best 0.056) |
| 9 | 11 | 0.484 | 1.54 | 0.32 | 0.17 | 0.19 | 0.16 | 0.16 | 0.79 | - | none (best 0.0942) |
| 10 | -122 | -0.54 | 1.4 | 0.15 | 0.14 | 0.21 | 0.17 | 0.33 | 0.582 | + | none (best 0.0693) |
| 11 | -111 | 0.499 | 1.63 | 0.30 | 0.18 | 0.22 | 0.18 | 0.13 | 0.434 | - | none (best 0.0516) |
| 12 | 132 | 1.67 | 5.25 | 0.01 | 0.42 | 0.56 | 0.01 | 0.00 | 0.337 | - | none (best 0.0209) |
| 13 | 34 | -0.602 | 1.34 | 0.15 | 0.13 | 0.20 | 0.17 | 0.35 | 0.63 | + | none (best 0.0751) |
| 14 | 137 | -0.58 | 1.75 | 0.10 | 0.15 | 0.27 | 0.22 | 0.26 | 1.01 | + | none (best 0.12) |
| 15 | 91 | -0.541 | 1.32 | 0.19 | 0.14 | 0.17 | 0.14 | 0.37 | 0.481 | + | none (best 0.0572) |

## Groups

- ring: 3 units (2, 9, 11) with gates at distance 0.48-0.52, switch width ~1.5, magnitude ~0.65 each, negative
- same line: units 0, 12 (gate angle -39 degrees, offset -1.57)
- the rest: units 1, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15
