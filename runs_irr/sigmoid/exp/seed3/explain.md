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
    return 1.056 * exp(0.9739 * x0) - 0.04886
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.4 +0.4 +0.2 +0.1 -0.0 -0.2 -0.3 -0.4 -0.3 +0.1
  - largest error per slice: +0.5 +0.4 +0.3 +0.2 +0.1 +0.3 +0.4 +0.4 +0.4 +0.4
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: -0.5 -0.4 -0.3 -0.1 +0.0 +0.1 +0.2 +0.4 +0.3 -0.2
  - largest error per slice: +0.5 +0.5 +0.4 +0.2 +0.1 +0.1 +0.3 +0.5 +0.5 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.056 | 5.76% |
| x0 inside exp | 1 | 0.9739 | 5.23% |
| offset | 0 | -0.04886 | 0.68% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -2.06 | 1.3 | 0.00 | 0.00 | 0.10 | 0.20 | 0.69 | 0.92 | - | none (best 0.114) |
| 1 | 0 | -10.8 | 20.4 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.145 | + | rounded hinge |
| 2 | 0 | 0.916 | 2.65 | 0.07 | 0.41 | 0.50 | 0.02 | 0.00 | 1.73 | + | none (best 0.112) |
| 3 | 0 | -10.1 | 18.8 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.163 | + | rounded hinge |
| 4 | 0 | 2.06 | 1.73 | 0.58 | 0.27 | 0.14 | 0.00 | 0.00 | 1.26 | + | none (best 0.144) |
| 5 | 180 | -2.07 | 1.35 | 0.00 | 0.00 | 0.11 | 0.21 | 0.68 | 0.928 | - | none (best 0.101) |
| 6 | 0 | 1.87 | 2.12 | 0.44 | 0.33 | 0.23 | 0.00 | 0.00 | 1.31 | + | none (best 0.129) |
| 7 | 0 | 0.757 | 2.74 | 0.00 | 0.43 | 0.52 | 0.05 | 0.00 | 0.727 | + | none (best 0.0449) |

## Groups

- same line: units 0, 4, 5, 6 (gate angle 180 degrees, offset -2.06)
- same line: units 2, 7 (gate angle 0 degrees, offset 0.916)
- the rest: units 1, 3
