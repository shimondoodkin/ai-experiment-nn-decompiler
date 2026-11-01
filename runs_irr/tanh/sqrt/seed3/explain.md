# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.01208)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.029 * sqrt(x0 + 0.02975) - 0.0519
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01208)
```

% of the output range:
- distortion (clean minus faithful): largest 1.1%, inside the box (x0 = 0.288)
  - mean error in 10 equal slices of x0: +0.9 +1.0 +0.7 +0.5 +0.3 +0.1 -0.0 -0.2 -0.4 -0.5
  - largest error per slice: +1.1 +1.0 +0.8 +0.6 +0.4 +0.2 +0.1 +0.3 +0.4 +0.6
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +0.2 -0.8 -0.4 -0.2 -0.1 +0.1 +0.2 +0.3 +0.4 +0.7
  - largest error per slice: +0.9 +0.9 +0.7 +0.2 +0.1 +0.1 +0.3 +0.4 +0.5 +0.9

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.01208 | 0.02975 | 3.37% |
| scale | 1 | 1.029 | 3.13% |
| offset | 0 | -0.0519 | 2.80% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0371, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.343 | 1.12 | 0.80 | 0.18 | 0.02 | 0.00 | 0.00 | 0.262 | - | none (best 0.0351) |
| 1 | 180 | -3.21 | 5.9 | 0.00 | 0.00 | 0.75 | 0.25 | 0.00 | 0.12 | - | rounded hinge |
| 2 | 180 | -0.823 | 9.4 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0867 | - | rounded hinge |
| 3 | 180 | -0.921 | 2.06 | 0.25 | 0.32 | 0.39 | 0.04 | 0.00 | 0.315 | - | none (best 0.0204) |
| 4 | 180 | 2.44 | 8.79 | 0.00 | 0.79 | 0.21 | 0.00 | 0.00 | 0.0853 | - | rounded hinge |
| 5 | 180 | 0.333 | 1.04 | 0.82 | 0.16 | 0.01 | 0.00 | 0.00 | 0.239 | - | none (best 0.0344) |
| 6 | 0 | 2.46 | 6.61 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.739 | + | none (best 0.0231) |
| 7 | 0 | 2.45 | 13.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00894 | + | constant |

## Groups

- negligible: units 7 (magnitude 0.00894 together; a quarter of the tolerance is 0.00927)
- same line: units 0, 5 (gate angle 180 degrees, offset 0.343)
- same line: units 2, 3 (gate angle 180 degrees, offset -0.823)
- the rest: units 1, 4, 6
