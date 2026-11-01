# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 2.7% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.7%, at the box edge (x0 = -0.0277, x1 = -3)
- by distance from (0, 0), rings 0.423 wide, mean: +0.2 +0.4 +0.1 -0.4 -0.1 +0.3 +0.2 -0.1 -0.3 -0.5; largest: +0.8 +1.0 +0.9 +1.6 +2.5 +2.6 +2.5 +2.7 +2.5 +2.6
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 -0.1 -0.2 +0.0 +0.2 +0.3 +0.2 +0.2 +0.3 -0.0 -0.4 -0.1; largest: +1.6 +2.6 +1.7 +2.5 +2.2 +2.6 +2.1 +2.0 +2.7 +2.6 +2.5 +2.4

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.9943 * exp(-0.9977 * x0 * x0 - 0.9837 * x1 * x1 - 0.003587 * x0 * x1 + 0.01187 * x0 + 0.000438 * x1) - 0.0002258
```

Clean:

```python
def net(x0, x1):
    return 0.9943 * exp(-0.9977 * x0 * x0 - 0.9837 * x1 * x1 - 0.003587 * x0 * x1 + 0.01187 * x0 + 0.000438 * x1) - 0.0002258
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.00595, 0.000212), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.7%, at the box edge (x0 = -0.0277, x1 = -3)
  - by distance from (0.00595, 0.000212), rings 0.424 wide, mean: -0.3 +0.3 +0.2 -0.3 -0.1 +0.3 +0.2 -0.1 -0.3 -0.5; largest: +0.8 +1.0 +1.0 +1.4 +2.5 +2.5 +2.5 +2.7 +2.5 +2.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 -0.1 -0.1 +0.1 +0.2 +0.2 +0.0 +0.1 +0.3 +0.1 -0.4 -0.0; largest: +1.7 +2.6 +1.7 +2.5 +2.2 +2.5 +2.1 +2.0 +2.7 +2.6 +2.5 +2.5

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0206, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -165 | 0.509 | 1.56 | 0.32 | 0.17 | 0.20 | 0.17 | 0.14 | 0.694 | - | none (best 0.0828) |
| 1 | -132 | 0.7 | 2.87 | 0.12 | 0.30 | 0.42 | 0.16 | 0.01 | 0.345 | + | none (best 0.0414) |
| 2 | -114 | 1.41 | 2.23 | 0.35 | 0.25 | 0.29 | 0.11 | 0.00 | 0.265 | + | none (best 0.0319) |
| 3 | 4 | 0.529 | 1.19 | 0.39 | 0.13 | 0.15 | 0.13 | 0.21 | 0.26 | - | none (best 0.0309) |
| 4 | 74 | 0.503 | 1.4 | 0.35 | 0.15 | 0.18 | 0.15 | 0.17 | 0.65 | - | none (best 0.0774) |
| 5 | 29 | 0.919 | 2.85 | 0.15 | 0.32 | 0.38 | 0.14 | 0.00 | 0.641 | + | none (best 0.0773) |
| 6 | 167 | 1.35 | 2.51 | 0.30 | 0.27 | 0.32 | 0.11 | 0.00 | 0.473 | + | none (best 0.0484) |
| 7 | 167 | -0.517 | 1.39 | 0.17 | 0.15 | 0.18 | 0.15 | 0.35 | 0.514 | + | none (best 0.0611) |
| 8 | 114 | 0.517 | 1.38 | 0.34 | 0.16 | 0.19 | 0.15 | 0.16 | 0.588 | - | none (best 0.0701) |
| 9 | 95 | -0.512 | 1.44 | 0.17 | 0.15 | 0.18 | 0.15 | 0.34 | 0.688 | + | none (best 0.082) |
| 10 | -47 | 1.07 | 2.79 | 0.18 | 0.33 | 0.37 | 0.12 | 0.00 | 0.496 | + | none (best 0.0597) |
| 11 | -47 | 0.523 | 1.37 | 0.32 | 0.18 | 0.21 | 0.13 | 0.15 | 0.583 | - | none (best 0.0692) |
| 12 | -145 | -0.543 | 1.39 | 0.15 | 0.14 | 0.21 | 0.18 | 0.33 | 0.668 | + | none (best 0.0796) |
| 13 | 100 | 1.28 | 2.65 | 0.27 | 0.28 | 0.33 | 0.11 | 0.00 | 0.394 | + | none (best 0.0361) |
| 14 | -127 | 0.537 | 1.43 | 0.31 | 0.19 | 0.22 | 0.14 | 0.14 | 0.671 | - | none (best 0.0799) |
| 15 | 151 | 0.532 | 1.54 | 0.31 | 0.19 | 0.22 | 0.16 | 0.13 | 0.683 | - | none (best 0.0813) |

## Groups

- ring: 6 units (0, 4, 8, 11, 14, 15) with gates at distance 0.50-0.54, switch width ~1.4, magnitude ~0.66 each, negative
- the rest: units 1, 2, 3, 5, 6, 7, 9, 10, 12, 13
