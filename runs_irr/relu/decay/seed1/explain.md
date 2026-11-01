# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 23 / (x0 + 4) - 2.736
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 20.37 / (x0 + 3.756) - 2.436
```

Clean:

```python
def net(x0):
    return 23 / (x0 + 4) - 2.736
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = 3)
  - mean error in 10 equal slices of x0: -0.1 +0.4 +0.6 +0.6 +0.5 +0.3 +0.1 -0.1 -0.4 -0.6
  - largest error per slice: +0.6 +0.6 +0.6 +0.6 +0.6 +0.4 +0.2 +0.2 +0.5 +0.7
- ripple (faithful minus net): largest 1.4%, inside the box (x0 = 0.652)
  - mean error in 10 equal slices of x0: -0.5 -0.7 -0.4 +0.1 +0.9 +0.4 +0.9 +0.0 -0.9 -1.3
  - largest error per slice: +1.4 +1.4 +0.6 +1.1 +1.4 +0.5 +1.3 +0.8 +1.2 +1.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 23 | 20.37 | 20.56% |
| constant inside reciprocal | 4 | 3.756 | 15.92% |
| offset | -2.736 | -2.436 | 7.04% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0853, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -9.33 | 0 | 0.00 | 1.00 | 0.374 | + | line |
| 1 | 180 | 0.217 | 0 | 0.80 | 0.20 | 1.07 | + | hinge |
| 2 | 180 | -1.68 | 0 | 0.33 | 0.67 | 1 | + | hinge |
| 3 | 0 | -6.89 | 0 | 0.00 | 1.00 | 0.458 | + | line |
| 4 | 180 | -0.651 | 0 | 0.59 | 0.41 | 0.844 | + | hinge |
| 5 | 0 | -0.206 | 0 | 0.20 | 0.80 | 1.72 | - | hinge |
| 6 | 180 | 2.34 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 7 | 0 | -0.206 | 0 | 0.20 | 0.80 | 0.457 | - | hinge |

## Groups

- negligible: units 6 (magnitude 0 together; a quarter of the tolerance is 0.0213)
- lines: units 0, 3 (each a constant or a line on the box; together one linear term)
- same line: units 1, 5, 7 (gate angle 180 degrees, offset 0.217)
- the rest: units 2, 4
