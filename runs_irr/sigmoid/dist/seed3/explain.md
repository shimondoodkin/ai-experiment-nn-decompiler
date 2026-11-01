# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (24 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 0.125) - 0.05
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.003 * x0 * x0 + 0.9886 * x1 * x1 - 0.004695 * x0 * x1 - 0.003339 * x0 - 0.02636 * x1 + 0.08256) - 0.03323
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 0.125) - 0.05
```

% of the output range:
- distortion (clean minus faithful): largest 1.3%, inside the box (x0 = 0.0026, x1 = 0.07)
  - by distance from (0, 0), rings 0.423 wide, mean: +0.9 +0.4 +0.1 +0.0 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1; largest: +1.3 +0.9 +0.6 +0.5 +0.5 +0.5 +0.5 +0.6 +0.5 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 +0.3 +0.5 +0.4 +0.1 -0.1 -0.3 -0.2 -0.1 -0.2 -0.3 -0.2; largest: +1.3 +1.3 +1.3 +1.3 +1.3 +1.3 +1.2 +1.3 +1.2 +1.2 +1.2 +1.3
- ripple (faithful minus net): largest 1.7%, inside the box (x0 = -0.632, x1 = -0.494)
  - by distance from (0.0017, 0.0133), rings 0.424 wide, mean: -0.3 +1.1 +0.5 -0.2 -0.4 -0.4 -0.5 -0.6 -0.6 -0.0; largest: +1.6 +1.7 +1.6 +1.3 +1.6 +1.6 +1.7 +1.7 +1.6 +1.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.5 -0.7 -0.6 -0.5 -0.1 -0.1 -0.2 -0.2 -0.3 -0.1 -0.1; largest: +1.6 +1.6 +1.6 +1.6 +1.6 +1.7 +1.6 +1.7 +1.6 +1.6 +1.6 +1.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.125 | 0.08256 | 1.70% |
| x1² inside sqrt | 1 | 0.9886 | 0.44% |
| offset | -0.05 | -0.03323 | 0.43% |
| x1 inside sqrt | 0 | -0.02636 | 0.34% |
| x0·x1 inside sqrt | 0 | -0.004695 | 0.13% |
| x0² inside sqrt | 1 | 1.003 | 0.11% |
| x0 inside sqrt | 0 | -0.003339 | 0.04% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0777, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 169 | 2.56 | 2.07 | 0.58 | 0.22 | 0.20 | 0.00 | 0.00 | 1.2 | + | none (best 0.0898) |
| 1 | -117 | 2.37 | 3.01 | 0.38 | 0.35 | 0.26 | 0.01 | 0.00 | 1.34 | + | none (best 0.0898) |
| 2 | -67 | 2.39 | 2.88 | 0.41 | 0.33 | 0.25 | 0.01 | 0.00 | 1.26 | + | none (best 0.0856) |
| 3 | 4 | 0.93 | 1.63 | 0.38 | 0.17 | 0.21 | 0.17 | 0.07 | 0.74 | + | none (best 0.0883) |
| 4 | 89 | 0.623 | 1.28 | 0.39 | 0.13 | 0.16 | 0.13 | 0.18 | 0.637 | + | none (best 0.0759) |
| 5 | 102 | 2.79 | 4.3 | 0.24 | 0.46 | 0.30 | 0.00 | 0.00 | 0.631 | + | none (best 0.0497) |
| 6 | 171 | 0.917 | 1.69 | 0.37 | 0.18 | 0.21 | 0.18 | 0.06 | 0.808 | + | none (best 0.0963) |
| 7 | -101 | 0.664 | 6.01 | 0.00 | 0.23 | 0.75 | 0.03 | 0.00 | 0.00465 | + | constant |
| 8 | 84 | 2.93 | 2.84 | 0.52 | 0.30 | 0.19 | 0.00 | 0.00 | 0.439 | + | none (best 0.0422) |
| 9 | 29 | 0.727 | 1.47 | 0.36 | 0.18 | 0.21 | 0.14 | 0.11 | 0.615 | + | none (best 0.0731) |
| 10 | -59 | 0.679 | 1.57 | 0.32 | 0.19 | 0.23 | 0.14 | 0.11 | 0.793 | + | none (best 0.0942) |
| 11 | 8 | 2.55 | 1.88 | 0.61 | 0.20 | 0.19 | 0.00 | 0.00 | 1.17 | + | none (best 0.0924) |
| 12 | -99 | -1.73 | 34.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.013 | - | constant |
| 13 | 146 | 0.656 | 1.23 | 0.38 | 0.16 | 0.18 | 0.12 | 0.16 | 0.416 | + | none (best 0.0494) |
| 14 | -123 | 0.682 | 1.58 | 0.32 | 0.20 | 0.23 | 0.15 | 0.11 | 0.801 | + | none (best 0.0955) |
| 15 | 88 | 1.88 | 2.14 | 0.46 | 0.22 | 0.27 | 0.05 | 0.00 | 0.627 | + | none (best 0.0415) |

## Groups

- negligible: units 7, 12 (magnitude 0.0177 together; a quarter of the tolerance is 0.0194)
- ring: 5 units (4, 9, 10, 13, 14) with gates at distance 0.62-0.73, switch width ~1.5, magnitude ~0.64 each, positive
- ring: 5 units (0, 1, 2, 5, 11) with gates at distance 2.37-2.79, switch width ~2.9, magnitude ~1.2 each, positive
- the rest: units 3, 6, 8, 15
