# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return exp(x0)
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0):
    return 1.051 * exp(0.977 * x0) - 0.048
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.4 +0.4 +0.3 +0.2 +0.0 -0.1 -0.3 -0.4 -0.3 -0.0
  - largest error per slice: +0.5 +0.4 +0.3 +0.2 +0.1 +0.2 +0.3 +0.4 +0.4 +0.3
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.4 -0.4 -0.4 -0.2 +0.0 +0.1 +0.1 +0.3 +0.3 -0.2
  - largest error per slice: +0.4 +0.4 +0.4 +0.3 +0.1 +0.1 +0.2 +0.5 +0.5 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.051 | 5.23% |
| x0 inside exp | 1 | 0.977 | 4.61% |
| offset | 0 | -0.048 | 0.67% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.94 | 2.03 | 0.48 | 0.32 | 0.20 | 0.00 | 0.00 | 0.811 | + | none (best 0.0814) |
| 1 | 0 | 0.753 | 2.12 | 0.16 | 0.33 | 0.40 | 0.11 | 0.00 | 1.63 | + | none (best 0.0972) |
| 2 | 0 | 0.281 | 5.75 | 0.00 | 0.03 | 0.97 | 0.00 | 0.00 | 0.897 | + | rounded hinge |
| 3 | 0 | 1.85 | 2.22 | 0.40 | 0.35 | 0.25 | 0.00 | 0.00 | 0.892 | + | none (best 0.0837) |
| 4 | 180 | -2.06 | 1.3 | 0.00 | 0.00 | 0.10 | 0.20 | 0.69 | 0.5 | - | none (best 0.0612) |
| 5 | 0 | 2.07 | 1.58 | 0.62 | 0.25 | 0.13 | 0.00 | 0.00 | 0.749 | + | none (best 0.0881) |
| 6 | 0 | 2.08 | 1.56 | 0.63 | 0.25 | 0.12 | 0.00 | 0.00 | 0.932 | + | none (best 0.106) |
| 7 | 0 | 2.07 | 1.31 | 0.69 | 0.21 | 0.10 | 0.00 | 0.00 | 0.795 | + | none (best 0.0983) |

## Groups

- same line: units 0, 3, 4, 5, 6, 7 (gate angle 0 degrees, offset 1.94)
- the rest: units 1, 2
