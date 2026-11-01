# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 2.9% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.9%, at the box edge (x0 = 3, x1 = -2.99)
- by distance from (0, 0), rings 0.423 wide, mean: -0.6 +0.4 +0.1 -0.5 -0.1 +0.3 +0.1 -0.2 -0.2 -0.1; largest: +1.2 +1.2 +1.1 +1.5 +1.9 +2.1 +1.8 +2.5 +2.5 +2.9
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.5 -0.6 -0.6 -0.3 +0.0 +0.5 +0.7 +0.6 +0.3 +0.1 -0.2 -0.3; largest: +2.5 +2.3 +1.5 +1.9 +2.4 +1.8 +2.1 +2.2 +1.8 +1.1 +2.9 +1.4

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.001 * exp(-1.047 * x0 * x0 - 1.004 * x1 * x1 - 0.02207 * x0 * x1 + 0.0466 * x0 + 0.01573 * x1) + 0.002001
```

Clean:

```python
def net(x0, x1):
    return 1.001 * exp(-1.047 * x0 * x0 - 1.004 * x1 * x1 - 0.02207 * x0 * x1 + 0.0466 * x0 + 0.01573 * x1) + 0.002001
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.0222, 0.00759), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.7%, at the box edge (x0 = 3, x1 = -2.99)
  - by distance from (0.0222, 0.00759), rings 0.424 wide, mean: -0.5 +0.0 -0.5 -0.9 -0.2 +0.4 +0.3 +0.1 +0.0 +0.2; largest: +1.9 +2.2 +2.2 +2.1 +1.7 +2.1 +2.0 +2.3 +2.3 +2.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.3 -0.4 -0.3 -0.1 +0.0 -0.0 +0.1 +0.3 +0.2 +0.3 +0.1 -0.1; largest: +2.3 +2.1 +1.4 +1.7 +2.6 +2.0 +2.2 +2.4 +1.9 +1.2 +2.7 +1.4

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0207, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -173 | 0.486 | 1.52 | 0.33 | 0.16 | 0.19 | 0.16 | 0.17 | 0.619 | - | none (best 0.0737) |
| 1 | -107 | 1.11 | 2.71 | 0.22 | 0.29 | 0.35 | 0.13 | 0.00 | 0.504 | + | none (best 0.0549) |
| 2 | -78 | 0.47 | 1.39 | 0.34 | 0.15 | 0.18 | 0.15 | 0.18 | 0.562 | - | none (best 0.067) |
| 3 | 22 | 0.438 | 1.43 | 0.32 | 0.16 | 0.19 | 0.16 | 0.17 | 0.53 | - | none (best 0.063) |
| 4 | 85 | 0.465 | 1.44 | 0.34 | 0.15 | 0.18 | 0.15 | 0.18 | 0.627 | - | none (best 0.0747) |
| 5 | 43 | 1.28 | 2.8 | 0.21 | 0.35 | 0.35 | 0.10 | 0.00 | 0.521 | + | none (best 0.063) |
| 6 | -178 | 1.19 | 2.91 | 0.21 | 0.30 | 0.36 | 0.12 | 0.00 | 0.511 | + | none (best 0.0302) |
| 7 | 170 | -0.452 | 1.39 | 0.19 | 0.15 | 0.18 | 0.15 | 0.34 | 0.568 | + | none (best 0.0675) |
| 8 | 120 | 0.457 | 1.48 | 0.30 | 0.18 | 0.22 | 0.15 | 0.15 | 0.679 | - | none (best 0.0806) |
| 9 | 37 | -0.416 | 1.44 | 0.16 | 0.15 | 0.22 | 0.18 | 0.29 | 0.567 | + | none (best 0.0675) |
| 10 | -70 | -0.687 | 3.39 | 0.00 | 0.15 | 0.45 | 0.34 | 0.06 | 0.651 | - | none (best 0.0688) |
| 11 | -44 | 0.481 | 1.46 | 0.29 | 0.19 | 0.23 | 0.14 | 0.15 | 0.538 | - | none (best 0.0641) |
| 12 | -128 | -0.459 | 1.59 | 0.13 | 0.15 | 0.25 | 0.20 | 0.27 | 0.646 | + | none (best 0.077) |
| 13 | 138 | -1.46 | 2.52 | 0.00 | 0.09 | 0.29 | 0.33 | 0.28 | 0.443 | - | none (best 0.0534) |
| 14 | -113 | 0.492 | 1.51 | 0.31 | 0.17 | 0.20 | 0.17 | 0.14 | 0.664 | - | none (best 0.0789) |
| 15 | 153 | 0.448 | 1.47 | 0.31 | 0.18 | 0.21 | 0.16 | 0.15 | 0.62 | - | none (best 0.0739) |

## Groups

- ring: 3 units (1, 5, 6) with gates at distance 1.11-1.28, switch width ~2.8, magnitude ~0.51 each, positive
- ring: 8 units (0, 2, 3, 4, 8, 11, 14, 15) with gates at distance 0.44-0.49, switch width ~1.5, magnitude ~0.62 each, negative
- the rest: units 7, 9, 10, 12, 13
