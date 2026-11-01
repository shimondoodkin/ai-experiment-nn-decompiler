# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (37 nodes):

```python
def net(x0):
    return -1.5 * x0 + 1.394 * relu(x0 - 0.9359) + 0.5 * relu(1.8 - x0) + 2
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return -0.4788 * x0 + 1.775
```

Clean:

```python
def net(x0):
    return -0.4788 * x0 + 1.775
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 22.7%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -5.5 +18.5 +21.2 +22.2 +18.3 +10.9 +3.4 -4.0 -11.5 -19.0
  - largest error per slice: +22.6 +20.7 +21.7 +22.7 +22.1 +14.6 +7.1 +7.8 +15.2 +22.7

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0335, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.19 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 180 | -1.87 | 0 | 0.61 | 0.39 | 0.557 | + | hinge |
| 2 | 180 | 0.229 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 180 | -0.971 | 0 | 0.86 | 0.14 | 0.619 | + | hinge |
| 4 | 180 | 12.3 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | 0.224 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 0 | 0.714 | 0 | 0.06 | 0.94 | 1.11 | + | hinge |
| 7 | 0 | 0.501 | 0 | 0.00 | 1.00 | 1.61 | - | line |

## Groups

- negligible: units 0, 2, 4, 5 (magnitude 0 together; a quarter of the tolerance is 0.00837)
- lines: units 7 (each a constant or a line on the box; together one linear term)
- the rest: units 1, 3, 6
