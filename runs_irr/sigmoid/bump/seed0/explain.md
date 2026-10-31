# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 2.6% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.6%, at the box edge (x0 = -2.99, x1 = -1.92)
- by distance from (0, 0), rings 0.423 wide, mean: +0.1 +0.2 -0.1 -0.6 -0.2 +0.2 +0.0 -0.1 -0.2 -0.4; largest: +0.2 +0.4 +0.9 +1.8 +1.9 +1.9 +1.7 +2.4 +2.6 +2.4
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 +0.1 -0.2 -0.5 -0.2 -0.1 -0.3 -0.3 +0.1 +0.2 +0.0 -0.1; largest: +2.3 +2.4 +2.0 +1.9 +2.4 +2.2 +2.5 +2.6 +2.6 +2.0 +2.4 +1.6

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.9933 * exp(-0.9875 * x0 * x0 - 0.9893 * x1 * x1 + 0.0001069 * x0 * x1 - 0.003932 * x0 + 0.002929 * x1) + 0.0008555
```

Clean:

```python
def net(x0, x1):
    return 0.9933 * exp(-0.9875 * x0 * x0 - 0.9893 * x1 * x1 + 0.0001069 * x0 * x1 - 0.003932 * x0 + 0.002929 * x1) + 0.0008555
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00199, 0.00148), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.5%, at the box edge (x0 = 2.46, x1 = -2.99)
  - by distance from (-0.00199, 0.00148), rings 0.424 wide, mean: -0.3 +0.2 +0.1 -0.3 -0.0 +0.3 +0.1 -0.1 -0.2 -0.3; largest: +0.5 +0.6 +0.6 +1.5 +2.0 +2.0 +1.8 +2.3 +2.5 +2.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.2 -0.0 -0.3 -0.0 +0.1 -0.1 -0.1 +0.2 +0.3 +0.1 +0.0; largest: +2.4 +2.5 +2.0 +1.8 +2.3 +2.1 +2.4 +2.5 +2.5 +1.9 +2.5 +1.7

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0205, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 107 | 0.51 | 1.45 | 0.34 | 0.16 | 0.19 | 0.16 | 0.16 | 0.702 | - | none (best 0.0835) |
| 1 | -133 | 0.537 | 1.38 | 0.32 | 0.18 | 0.21 | 0.13 | 0.15 | 0.589 | - | none (best 0.0702) |
| 2 | -60 | 1.27 | 2.49 | 0.26 | 0.30 | 0.32 | 0.11 | 0.00 | 0.444 | + | none (best 0.0535) |
| 3 | 154 | 1.16 | 2.68 | 0.22 | 0.31 | 0.35 | 0.12 | 0.00 | 0.533 | + | none (best 0.0645) |
| 4 | -133 | 0.956 | 3.15 | 0.11 | 0.33 | 0.43 | 0.12 | 0.00 | 0.134 | + | none (best 0.0161) |
| 5 | -152 | -0.54 | 1.43 | 0.14 | 0.15 | 0.20 | 0.17 | 0.33 | 0.662 | + | none (best 0.0788) |
| 6 | 166 | -0.514 | 1.43 | 0.17 | 0.15 | 0.19 | 0.15 | 0.34 | 0.691 | + | none (best 0.0823) |
| 7 | -97 | -1.12 | 2.73 | 0.00 | 0.14 | 0.34 | 0.29 | 0.23 | 0.48 | - | none (best 0.0411) |
| 8 | 87 | -0.51 | 1.45 | 0.17 | 0.15 | 0.18 | 0.15 | 0.34 | 0.706 | + | none (best 0.0842) |
| 9 | -173 | 0.498 | 1.48 | 0.33 | 0.15 | 0.19 | 0.15 | 0.17 | 0.737 | - | none (best 0.0878) |
| 10 | 66 | 0.523 | 1.42 | 0.34 | 0.16 | 0.19 | 0.16 | 0.15 | 0.666 | - | none (best 0.0793) |
| 11 | 146 | 0.547 | 1.43 | 0.32 | 0.18 | 0.21 | 0.14 | 0.14 | 0.661 | - | none (best 0.0786) |
| 12 | -131 | 1.25 | 2.61 | 0.23 | 0.33 | 0.33 | 0.11 | 0.00 | 0.37 | + | none (best 0.0446) |
| 13 | 36 | 0.53 | 4.5 | 0.00 | 0.27 | 0.62 | 0.11 | 0.00 | 0.0502 | + | rounded hinge |
| 14 | -167 | -1.27 | 2.47 | 0.00 | 0.12 | 0.32 | 0.27 | 0.29 | 0.448 | - | none (best 0.0471) |
| 15 | 127 | -0.542 | 1.38 | 0.15 | 0.13 | 0.21 | 0.18 | 0.32 | 0.595 | + | none (best 0.0707) |

## Groups

- same line: units 4, 12 (gate angle -133 degrees, offset 0.956)
- same line: units 5, 13 (gate angle -152 degrees, offset -0.54)
- the rest: units 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 15
