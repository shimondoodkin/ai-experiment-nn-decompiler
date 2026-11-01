# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.01196)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.028 * sqrt(x0 + 0.02921) - 0.05056
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01196)
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, inside the box (x0 = 0.292)
  - mean error in 10 equal slices of x0: +0.9 +0.9 +0.7 +0.5 +0.3 +0.2 -0.0 -0.2 -0.3 -0.5
  - largest error per slice: +1.0 +1.0 +0.8 +0.6 +0.4 +0.2 +0.1 +0.2 +0.4 +0.5
- ripple (faithful minus net): largest 1.0%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: +0.2 -0.9 -0.4 -0.1 -0.0 +0.0 +0.1 +0.2 +0.4 +0.8
  - largest error per slice: +1.0 +0.9 +0.7 +0.2 +0.0 +0.0 +0.1 +0.3 +0.6 +1.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.01196 | 0.02921 | 3.33% |
| scale | 1 | 1.028 | 3.03% |
| offset | 0 | -0.05056 | 2.73% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.037, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.14 | 3.48 | 0.09 | 0.55 | 0.36 | 0.00 | 0.00 | 0.47 | - | none (best 0.0391) |
| 1 | 0 | -7.33 | 27.2 | 0.00 | 0.00 | 0.71 | 0.29 | 0.00 | 0.0388 | + | rounded hinge |
| 2 | 0 | 0.2 | 10.7 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.224 | + | rounded hinge |
| 3 | 180 | -3.26 | 16.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0165 | - | constant |
| 4 | 0 | 2.22 | 8.65 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.321 | + | rounded hinge |
| 5 | 180 | 0.222 | 0.917 | 0.82 | 0.15 | 0.03 | 0.00 | 0.00 | 0.388 | - | none (best 0.0536) |
| 6 | 0 | 2.48 | 7.9 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.392 | + | none (best 0.0102) |
| 7 | 0 | 3.15 | 37.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.000121 | - | constant |

## Groups

- negligible: units 7 (magnitude 0.000121 together; a quarter of the tolerance is 0.00925)
- lines: units 3 (each a constant or a line on the box; together one linear term)
- same line: units 0, 2 (gate angle 180 degrees, offset -0.14)
- the rest: units 1, 4, 5, 6
