# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (38 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 - 0.003092 * x0 + 0.05 * x1 + 0.1) - 0.03263
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.018 * x0 * x0 + 1.013 * x1 * x1 + 0.01242 * x0 * x1 + 0.01167 * x0 + 0.02645 * x1 + 0.1264) - 0.06545
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 - 0.003092 * x0 + 0.05 * x1 + 0.1) - 0.03263
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, inside the box (x0 = -0.947, x1 = 1.45)
  - by distance from (0.00155, -0.025), rings 0.425 wide, mean: +0.0 +0.3 +0.3 +0.3 +0.3 +0.2 +0.2 +0.1 +0.0 -0.0; largest: +0.4 +0.7 +0.8 +0.8 +0.8 +0.8 +0.8 +0.8 +0.7 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.1 +0.4 +0.7 +0.7 +0.5 +0.2 -0.1 -0.0 +0.0 +0.0 +0.0; largest: +0.2 +0.4 +0.7 +0.8 +0.8 +0.7 +0.5 +0.5 +0.3 +0.3 +0.3 +0.2
- ripple (faithful minus net): largest 1.9%, inside the box (x0 = 2.01, x1 = -2.48)
  - by distance from (-0.00565, -0.013), rings 0.424 wide, mean: -0.6 +1.0 +0.9 +0.1 -0.5 -0.7 -0.5 -0.3 -0.4 -0.1; largest: +1.8 +1.8 +1.8 +1.5 +1.7 +1.8 +1.8 +1.9 +1.7 +1.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 +0.0 +0.1 +0.0 -0.4 -0.2 -0.3 -0.4 -0.4 -0.3 -0.7 -0.1; largest: +1.8 +1.8 +1.8 +1.8 +1.8 +1.8 +1.8 +1.8 +1.8 +1.7 +1.9 +1.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.1 | 0.1264 | 1.01% |
| offset | -0.03263 | -0.06545 | 0.84% |
| x0² inside sqrt | 1 | 1.018 | 0.69% |
| x1² inside sqrt | 1 | 1.013 | 0.50% |
| x0·x1 inside sqrt | 0 | 0.01242 | 0.34% |
| x1 inside sqrt | 0.05 | 0.02645 | 0.30% |
| x0 inside sqrt | -0.003092 | 0.01167 | 0.19% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0774, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -81 | 0.629 | 1.19 | 0.40 | 0.13 | 0.15 | 0.13 | 0.19 | 0.303 | + | none (best 0.036) |
| 1 | -35 | -0.468 | 4.93 | 0.00 | 0.10 | 0.67 | 0.23 | 0.00 | 0.14 | - | clamp |
| 2 | -55 | 0.822 | 1.62 | 0.34 | 0.21 | 0.24 | 0.13 | 0.09 | 0.702 | + | none (best 0.0835) |
| 3 | -134 | 0.937 | 2.22 | 0.24 | 0.28 | 0.31 | 0.14 | 0.03 | 1.12 | + | none (best 0.134) |
| 4 | -96 | 1.65 | 8.35 | 0.00 | 0.25 | 0.75 | 0.00 | 0.00 | 0.108 | + | rounded hinge |
| 5 | 11 | 2.61 | 3.28 | 0.39 | 0.35 | 0.26 | 0.00 | 0.00 | 1.57 | + | none (best 0.113) |
| 6 | 77 | 0.722 | 2.03 | 0.28 | 0.22 | 0.26 | 0.20 | 0.04 | 1.19 | + | none (best 0.142) |
| 7 | 160 | 0.801 | 2.04 | 0.28 | 0.23 | 0.27 | 0.18 | 0.04 | 1.33 | + | none (best 0.159) |
| 8 | -78 | 2.45 | 2.41 | 0.50 | 0.26 | 0.23 | 0.00 | 0.00 | 1.46 | + | none (best 0.106) |
| 9 | 81 | 2.6 | 2.7 | 0.48 | 0.28 | 0.23 | 0.00 | 0.00 | 1.31 | + | none (best 0.101) |
| 10 | -72 | 1.29 | 3.09 | 0.18 | 0.34 | 0.39 | 0.08 | 0.00 | 0.12 | + | clamp |
| 11 | -98 | 1.29 | 10 | 0.00 | 0.08 | 0.92 | 0.00 | 0.00 | 0.107 | + | line |
| 12 | -76 | 0.827 | 1.8 | 0.33 | 0.19 | 0.23 | 0.18 | 0.06 | 0.285 | + | none (best 0.034) |
| 13 | -165 | -0.726 | 1.82 | 0.07 | 0.19 | 0.23 | 0.20 | 0.31 | 1.01 | - | none (best 0.121) |
| 14 | -68 | 0.271 | 6.15 | 0.00 | 0.14 | 0.79 | 0.07 | 0.00 | 0.018 | - | constant |
| 15 | 170 | 2.64 | 3.13 | 0.42 | 0.33 | 0.25 | 0.00 | 0.00 | 1.58 | + | none (best 0.116) |

## Groups

- negligible: units 14 (magnitude 0.018 together; a quarter of the tolerance is 0.0194)
- lines: units 11 (each a constant or a line on the box; together one linear term)
- ring: 3 units (2, 6, 7) with gates at distance 0.72-0.82, switch width ~2, magnitude ~1.2 each, positive
- ring: 4 units (5, 8, 9, 15) with gates at distance 2.45-2.64, switch width ~2.9, magnitude ~1.5 each, positive
- same line: units 0, 12 (gate angle -81 degrees, offset 0.629)
- the rest: units 1, 3, 4, 10, 13
