# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 17.5 / (x0 + 3.5) - 2
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 18.45 / (x0 + 3.588) - 2.166
```

Clean:

```python
def net(x0):
    return 17.5 / (x0 + 3.5) - 2
```

% of the output range:
- distortion (clean minus faithful): largest 1.4%, at the box edge (x0 = 3)
  - mean error in 10 equal slices of x0: +0.7 +0.6 +0.6 +0.6 +0.7 +0.8 +1.0 +1.1 +1.2 +1.3
  - largest error per slice: +0.9 +0.6 +0.6 +0.7 +0.8 +0.9 +1.0 +1.1 +1.3 +1.4
- ripple (faithful minus net): largest 1.1%, inside the box (x0 = -0.572)
  - mean error in 10 equal slices of x0: -0.5 -1.0 -0.5 +0.1 +0.6 +0.9 +1.0 +0.8 +0.3 -0.5
  - largest error per slice: +1.1 +1.1 +0.8 +0.4 +0.8 +1.0 +1.0 +1.0 +0.6 +1.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 17.5 | 18.45 | 8.96% |
| constant inside reciprocal | 3.5 | 3.588 | 5.61% |
| offset | -2 | -2.166 | 3.91% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0848, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -1.06 | 4.79 | 0.00 | 0.03 | 0.90 | 0.07 | 0.00 | 0.479 | + | rounded hinge |
| 1 | 180 | 0.409 | 2.39 | 0.25 | 0.38 | 0.37 | 0.00 | 0.00 | 0.684 | + | none (best 0.0498) |
| 2 | 180 | -0.841 | 4.59 | 0.00 | 0.11 | 0.86 | 0.03 | 0.00 | 0.755 | + | none (best 0.0253) |
| 3 | 180 | -0.888 | 4.8 | 0.00 | 0.08 | 0.90 | 0.02 | 0.00 | 0.522 | + | rounded hinge |
| 4 | 180 | 0.98 | 1.13 | 0.71 | 0.17 | 0.11 | 0.00 | 0.00 | 0.528 | + | none (best 0.0553) |
| 5 | 180 | 0.379 | 3.34 | 0.01 | 0.52 | 0.47 | 0.00 | 0.00 | 0.292 | + | rounded hinge |
| 6 | 180 | 0.7 | 2.09 | 0.40 | 0.33 | 0.27 | 0.00 | 0.00 | 0.589 | + | none (best 0.0522) |
| 7 | 180 | 0.13 | 3.68 | 0.00 | 0.44 | 0.56 | 0.00 | 0.00 | 0.393 | + | none (best 0.0259) |

## Groups

- same line: units 0, 2, 3 (gate angle 180 degrees, offset -1.06)
- same line: units 1, 5 (gate angle 180 degrees, offset 0.409)
- the rest: units 4, 6, 7
