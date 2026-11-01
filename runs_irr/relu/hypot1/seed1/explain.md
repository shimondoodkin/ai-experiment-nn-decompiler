# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (51 nodes):

```python
def net(x0, x1):
    u = 0.1239 * x0 * x0 + 0.125 * x1 * x1 + 0.5986
    return 2 * log(u) - 9 * relu(0.61 - u) + 2.051
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.557 * log(0.1616 * x0 * x0 + 0.1588 * x1 * x1 - 0.0002481 * x0 * x1 + 0.0008898 * x0 + 0.000101 * x1 + 0.4991) + 2.06
```

Clean:

```python
def net(x0, x1):
    return 1.557 * log(0.1616 * x0 * x0 + 0.1588 * x1 * x1 - 0.0002481 * x0 * x1 + 0.0008898 * x0 + 0.000101 * x1 + 0.4991) + 2.06
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.548, x1 = 0.428)
  - by distance from (-0.00275, -0.00032), rings 0.282 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.6%, inside the box (x0 = 0.0126, x1 = -0.0035)
  - by distance from (-0.00275, -0.00032), rings 0.282 wide, mean: -0.1 -1.6 -0.7 +0.1 +0.4 +0.8 +1.2 +1.2 +0.5 -0.5; largest: +2.6 +2.5 +2.4 +2.2 +2.2 +2.5 +2.3 +2.5 +2.4 +2.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.8 +0.4 +0.4 +0.4 +0.5 +0.7 +0.8 +0.4 +0.4 +0.4 +0.4 +0.7; largest: +2.5 +2.5 +2.6 +2.4 +2.4 +2.5 +2.5 +2.6 +2.3 +2.5 +2.5 +2.6

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0413, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | -0.00473 | 0 | 0.50 | 0.50 | 0.63 | + | hinge |
| 1 | 106 | 2.55 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 2 | 176 | 0.839 | 0 | 0.71 | 0.29 | 0.326 | + | hinge |
| 3 | 88 | 2.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 55 | -3.05 | 0 | 0.00 | 1.00 | 0.381 | + | line |
| 5 | -60 | 0.016 | 0 | 0.50 | 0.50 | 0.619 | + | hinge |
| 6 | -68 | -0.767 | 0 | 0.29 | 0.71 | 0.828 | + | hinge |
| 7 | -129 | 4.01 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | 112 | -0.782 | 0 | 0.29 | 0.71 | 0.794 | + | hinge |
| 9 | -4 | 0.838 | 0 | 0.71 | 0.29 | 0.331 | + | hinge |
| 10 | -125 | -0.64 | 0 | 0.30 | 0.70 | 0.835 | + | hinge |
| 11 | -124 | -0.0068 | 0 | 0.50 | 0.50 | 0.867 | + | hinge |
| 12 | -156 | 3.23 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 13 | 54 | -0.626 | 0 | 0.31 | 0.69 | 0.851 | + | hinge |
| 14 | 165 | 0.000242 | 0 | 0.50 | 0.50 | 0.65 | + | hinge |
| 15 | 103 | -0.00801 | 0 | 0.50 | 0.50 | 0.619 | + | hinge |

## Groups

- negligible: units 1, 3, 7, 12 (magnitude 0 together; a quarter of the tolerance is 0.0103)
- lines: units 4 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15
