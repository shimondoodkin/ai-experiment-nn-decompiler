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
    return 1.035 * exp(0.983 * x0) - 0.02863
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.2 +0.2 +0.1 +0.0 -0.0 -0.1 -0.2 -0.3 -0.2 +0.1
  - largest error per slice: +0.3 +0.2 +0.2 +0.1 +0.1 +0.2 +0.3 +0.3 +0.3 +0.4
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = 1.83)
  - mean error in 10 equal slices of x0: -0.3 -0.2 -0.1 -0.1 +0.0 +0.1 +0.2 +0.3 +0.2 -0.2
  - largest error per slice: +0.3 +0.2 +0.2 +0.1 +0.1 +0.1 +0.2 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.035 | 3.59% |
| x0 inside exp | 1 | 0.983 | 3.43% |
| offset | 0 | -0.02863 | 0.40% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.513 | 3.85 | 0.00 | 0.27 | 0.72 | 0.01 | 0.00 | 1.13 | + | none (best 0.0569) |
| 1 | 0 | 1.46 | 2.66 | 0.20 | 0.41 | 0.38 | 0.00 | 0.00 | 0.933 | + | none (best 0.0721) |
| 2 | 180 | -1.48 | 2.6 | 0.00 | 0.00 | 0.37 | 0.41 | 0.22 | 0.336 | - | rounded hinge |
| 3 | 180 | -1.59 | 2.5 | 0.00 | 0.00 | 0.34 | 0.39 | 0.27 | 0.251 | - | rounded hinge |
| 4 | 0 | 2.13 | 1.63 | 0.62 | 0.26 | 0.12 | 0.00 | 0.00 | 1.31 | + | none (best 0.155) |
| 5 | 0 | 1.64 | 2.57 | 0.27 | 0.40 | 0.33 | 0.00 | 0.00 | 1.19 | + | none (best 0.0985) |
| 6 | 0 | 2.07 | 1.06 | 0.75 | 0.17 | 0.08 | 0.00 | 0.00 | 1.17 | + | none (best 0.13) |
| 7 | 0 | 1.79 | 2.35 | 0.36 | 0.37 | 0.27 | 0.00 | 0.00 | 0.885 | + | none (best 0.0825) |

## Groups

- same line: units 1, 2, 3, 5, 7 (gate angle 0 degrees, offset 1.46)
- same line: units 4, 6 (gate angle 0 degrees, offset 2.13)
- the rest: units 0
