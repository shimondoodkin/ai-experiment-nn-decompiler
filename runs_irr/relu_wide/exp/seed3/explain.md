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
    return 1.004 * exp(0.9975 * x0) - 0.01212
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2
  - largest error per slice: +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.3
- ripple (faithful minus net): largest 0.2%, at the box edge (x0 = 1.92)
  - mean error in 10 equal slices of x0: -0.0 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.0 -0.1
  - largest error per slice: +0.0 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0 inside exp | 1 | 0.9975 | 0.51% |
| scale | 1 | 1.004 | 0.41% |
| offset | 0 | -0.01212 | 0.17% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.145, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.399 | 0 | 0.60 | 0.40 | 0.687 | - | hinge |
| 1 | 180 | 1.37 | 0 | 0.84 | 0.16 | 0.0498 | + | constant |
| 2 | 180 | 2.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | -0.192 | 0 | 0.45 | 0.55 | 0.34 | - | hinge |
| 4 | 180 | 4.63 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 1.68 | 0 | 0.92 | 0.08 | 0.018 | + | constant |
| 6 | 0 | 1.44 | 0 | 0.86 | 0.14 | 0.449 | + | hinge |
| 7 | 0 | 0.196 | 0 | 0.55 | 0.45 | 1.07 | + | hinge |
| 8 | 0 | 1.84 | 0 | 0.96 | 0.04 | 0.155 | + | hinge |
| 9 | 0 | 0.758 | 0 | 0.69 | 0.31 | 0.759 | + | hinge |
| 10 | 180 | 2.1 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 11 | 0 | 1.62 | 0 | 0.91 | 0.09 | 0.329 | + | hinge |
| 12 | 180 | -1.73 | 0 | 0.07 | 0.93 | 0.569 | + | hinge |
| 13 | 180 | 2.59 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 14 | 180 | 1.06 | 0 | 0.76 | 0.24 | 0.103 | + | hinge |
| 15 | 180 | 2.97 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 16 | 180 | -0.754 | 0 | 0.31 | 0.69 | 0.17 | - | hinge |
| 17 | 0 | -0.724 | 0 | 0.32 | 0.68 | 0.45 | + | hinge |
| 18 | 0 | -0.0902 | 0 | 0.48 | 0.52 | 0.545 | + | hinge |
| 19 | 0 | 1 | 0 | 0.75 | 0.25 | 0.625 | + | hinge |
| 20 | 0 | 0.488 | 0 | 0.62 | 0.38 | 0.843 | + | hinge |
| 21 | 180 | -1.34 | 0 | 0.16 | 0.84 | 0.119 | + | line |
| 22 | 0 | -0.398 | 0 | 0.40 | 0.60 | 1.54 | + | hinge |
| 23 | 180 | -0.485 | 0 | 0.38 | 0.62 | 0.255 | - | hinge |
| 24 | 180 | -0.193 | 0 | 0.45 | 0.55 | 0.173 | - | hinge |
| 25 | 180 | -1.73 | 0 | 0.07 | 0.93 | 0.142 | + | line |
| 26 | 180 | 1.82 | 0 | 0.96 | 0.04 | 6.85e-21 | + | constant |
| 27 | 0 | 2.23 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 180 | -1.12 | 0 | 0.22 | 0.78 | 0.0655 | + | constant |
| 29 | 180 | -1.73 | 0 | 0.07 | 0.93 | 0.632 | + | hinge |
| 30 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 31 | 0 | 1.23 | 0 | 0.81 | 0.19 | 0.547 | + | hinge |

## Groups

- negligible: units 2, 4, 5, 10, 13, 15, 26, 27, 30 (magnitude 0.018 together; a quarter of the tolerance is 0.0362)
- lines: units 1, 21, 25, 28 (each a constant or a line on the box; together one linear term)
- same line: units 0, 22 (gate angle 180 degrees, offset 0.399)
- same line: units 3, 7, 24 (gate angle 180 degrees, offset -0.192)
- same line: units 6, 8, 11, 12, 29 (gate angle 0 degrees, offset 1.44)
- same line: units 9, 16 (gate angle 0 degrees, offset 0.758)
- same line: units 20, 23 (gate angle 0 degrees, offset 0.488)
- the rest: units 14, 17, 18, 19, 31
