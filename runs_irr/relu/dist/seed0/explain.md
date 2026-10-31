# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (18 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.009 * x0 * x0 + 1.009 * x1 * x1 - 0.00474 * x0 * x1 + 0.01282 * x0 + 0.007068 * x1 + 0.0174) - 0.02691
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 2.3%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.2 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.2 +0.2 +0.2; largest: +2.3 +0.5 +0.6 +0.6 +0.6 +0.6 +0.5 +0.5 +0.5 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +0.2 +0.2 +0.2 +0.2 +0.4 +0.5 +0.5 +0.4 +0.3 +0.1 +0.1; largest: +1.9 +2.1 +1.6 +2.3 +1.8 +1.6 +1.7 +2.2 +1.4 +1.4 +1.4 +2.1
- ripple (faithful minus net): largest 1.0%, at the box edge (x0 = 0.0901, x1 = 2.98)
  - by distance from (-0.00636, -0.00352), rings 0.424 wide, mean: -0.2 -0.3 -0.3 -0.3 -0.3 -0.3 -0.3 -0.3 -0.2 -0.1; largest: +0.9 +0.7 +0.8 +0.8 +0.8 +0.8 +1.0 +1.0 +0.8 +0.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.3 -0.2 -0.2 -0.1 -0.2 -0.3 -0.4 -0.3 -0.4 -0.3 -0.3; largest: +0.8 +0.9 +1.0 +0.5 +0.7 +0.8 +0.8 +0.7 +0.9 +0.9 +0.8 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.0174 | 2.65% |
| offset | 0 | -0.02691 | 0.65% |
| x0² inside sqrt | 1 | 1.009 | 0.33% |
| x1² inside sqrt | 1 | 1.009 | 0.33% |
| x0 inside sqrt | 0 | 0.01282 | 0.16% |
| x0·x1 inside sqrt | 0 | -0.00474 | 0.12% |
| x1 inside sqrt | 0 | 0.007068 | 0.09% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.083, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 43 | 0.207 | 0 | 0.55 | 0.45 | 0.333 | + | hinge |
| 1 | -131 | 0.336 | 0 | 0.57 | 0.43 | 0.371 | + | hinge |
| 2 | 180 | 0.399 | 0 | 0.57 | 0.43 | 0.144 | + | hinge |
| 3 | 71 | -0.00861 | 0 | 0.50 | 0.50 | 1.45 | + | hinge |
| 4 | 133 | 0.00283 | 0 | 0.50 | 0.50 | 1.26 | + | hinge |
| 5 | -83 | -0.0857 | 0 | 0.48 | 0.52 | 1.26 | + | hinge |
| 6 | -129 | -0.0327 | 0 | 0.49 | 0.51 | 0.938 | + | hinge |
| 7 | -146 | 0.011 | 0 | 0.50 | 0.50 | 0.785 | + | hinge |
| 8 | -1 | 0.0445 | 0 | 0.51 | 0.49 | 1.24 | + | hinge |
| 9 | -111 | 1.18 | 0 | 0.71 | 0.29 | 0.0611 | - | hinge |
| 10 | 117 | 0.0108 | 0 | 0.50 | 0.50 | 1.02 | + | hinge |
| 11 | 88 | 3.11 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 151 | 0.523 | 0 | 0.60 | 0.40 | 0.0344 | + | constant |
| 13 | -27 | 0.015 | 0 | 0.50 | 0.50 | 1.63 | + | hinge |
| 14 | -159 | -0.00372 | 0 | 0.50 | 0.50 | 1.03 | + | hinge |
| 15 | 96 | -0.34 | 0 | 0.44 | 0.56 | 0.429 | + | hinge |

## Groups

- negligible: units 11 (magnitude 0 together; a quarter of the tolerance is 0.0208)
- lines: units 12 (each a constant or a line on the box; together one linear term)
- same line: units 0, 6, 7 (gate angle 43 degrees, offset 0.207)
- the rest: units 1, 2, 3, 4, 5, 8, 9, 10, 13, 14, 15
