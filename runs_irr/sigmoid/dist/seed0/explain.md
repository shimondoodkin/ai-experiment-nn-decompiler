# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (55 nodes):

```python
def net(x0, x1):
    return sqrt(1.03 * x0 * x0 + 1.034 * x1 * x1 + 0.01471 * x0 * x1 + 0.02 * x0 + 0.03 * x1 + 0.125) - 0.09478
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.03 * x0 * x0 + 1.034 * x1 * x1 + 0.01471 * x0 * x1 + 0.02612 * x0 + 0.03025 * x1 + 0.1285) - 0.09478
```

Clean:

```python
def net(x0, x1):
    return sqrt(1.03 * x0 * x0 + 1.034 * x1 * x1 + 0.01471 * x0 * x1 + 0.02 * x0 + 0.03 * x1 + 0.125) - 0.09478
```

% of the output range:
- distortion (clean minus faithful): largest 0.1%, inside the box (x0 = 0.204, x1 = 0.00337)
  - by distance from (-0.00961, -0.0144), rings 0.424 wide, mean: -0.1 -0.1 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 -0.1 -0.0 -0.0 +0.0 +0.0 +0.0 +0.0 -0.0 -0.0 -0.1 -0.1; largest: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 1.9%, inside the box (x0 = -1.81, x1 = 0.923)
  - by distance from (-0.0126, -0.0145), rings 0.423 wide, mean: -0.7 +0.6 +0.3 -0.5 -0.8 -0.8 -0.7 -0.4 -0.5 -0.2; largest: +1.8 +1.9 +1.8 +1.9 +1.9 +1.9 +1.9 +1.9 +1.6 +1.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.0 -0.0 -0.3 -0.8 -0.7 -0.7 -0.7 -0.7 -0.5 -0.8 -0.3; largest: +1.9 +1.9 +1.9 +1.7 +1.9 +1.9 +1.8 +1.8 +1.9 +1.8 +1.9 +1.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.125 | 0.1285 | 0.12% |
| x0 inside sqrt | 0.02 | 0.02612 | 0.07% |
| x1 inside sqrt | 0.03 | 0.03025 | 0.00% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0795, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 48 | 0.855 | 1.99 | 0.27 | 0.26 | 0.28 | 0.14 | 0.05 | 0.905 | + | none (best 0.108) |
| 1 | -99 | 2.41 | 2.43 | 0.49 | 0.26 | 0.25 | 0.00 | 0.00 | 1.01 | + | none (best 0.0767) |
| 2 | -168 | 2.62 | 2.91 | 0.45 | 0.31 | 0.24 | 0.00 | 0.00 | 1.49 | + | none (best 0.113) |
| 3 | 118 | 0.886 | 2.17 | 0.26 | 0.26 | 0.30 | 0.15 | 0.03 | 1.1 | + | none (best 0.131) |
| 4 | -9 | 2.35 | 2.45 | 0.48 | 0.26 | 0.25 | 0.00 | 0.00 | 1.66 | + | none (best 0.113) |
| 5 | -102 | -1.89 | 15.8 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00907 | - | constant |
| 6 | 66 | 0.222 | 11.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0904 | + | line |
| 7 | 174 | -0.657 | 1.18 | 0.19 | 0.12 | 0.15 | 0.13 | 0.41 | 0.519 | - | none (best 0.0616) |
| 8 | -31 | 0.803 | 1.74 | 0.32 | 0.21 | 0.25 | 0.14 | 0.08 | 0.754 | + | none (best 0.0896) |
| 9 | -161 | 0.704 | 1.94 | 0.28 | 0.21 | 0.26 | 0.18 | 0.06 | 1.09 | + | none (best 0.13) |
| 10 | 72 | 1.04 | 5.69 | 0.00 | 0.31 | 0.67 | 0.02 | 0.00 | 0.132 | + | clamp |
| 11 | 99 | 2.5 | 2.24 | 0.55 | 0.23 | 0.22 | 0.00 | 0.00 | 1.34 | + | none (best 0.0982) |
| 12 | -102 | 0.677 | 1.73 | 0.32 | 0.18 | 0.22 | 0.18 | 0.09 | 0.937 | + | none (best 0.112) |
| 13 | -94 | 3.58 | 5.67 | 0.15 | 0.59 | 0.26 | 0.00 | 0.00 | 0.421 | + | none (best 0.0396) |
| 14 | -81 | -0.694 | 1.35 | 0.15 | 0.14 | 0.17 | 0.14 | 0.39 | 0.457 | - | none (best 0.0544) |
| 15 | -80 | -5.63 | 91.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.000807 | + | constant |

## Groups

- negligible: units 5, 15 (magnitude 0.00988 together; a quarter of the tolerance is 0.0199)
- lines: units 6 (each a constant or a line on the box; together one linear term)
- ring: 5 units (0, 3, 8, 9, 12) with gates at distance 0.68-0.89, switch width ~1.9, magnitude ~0.94 each, positive
- ring: 4 units (1, 2, 4, 11) with gates at distance 2.35-2.62, switch width ~2.4, magnitude ~1.4 each, positive
- the rest: units 7, 10, 13, 14
