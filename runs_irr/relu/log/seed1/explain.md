# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (35 nodes):

```python
def net(x0):
    return 3 * x0 - 2.7 * relu(x0 - 0.5) - 0.6 * relu(1.8 - x0) - 1.25
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return 0.7506 * x0 - 1.382
```

Clean:

```python
def net(x0):
    return 0.7506 * x0 - 1.382
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0.1)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 19.0%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: +0.8 -15.1 -16.9 -18.2 -13.7 -7.8 -1.8 +4.1 +10.1 +16.1
  - largest error per slice: +19.0 +16.0 +17.9 +19.0 +16.7 +10.7 +4.8 +7.1 +13.1 +19.0

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0735, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0981 | 0 | 0.00 | 1.00 | 1.81 | + | line |
| 1 | 180 | 0.277 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | -2.3e-05 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 0 | 0.0978 | 0 | 0.00 | 1.00 | 1.96 | + | line |
| 4 | 180 | -0.537 | 0 | 0.91 | 0.09 | 1.2 | - | hinge |
| 5 | 0 | 1.87 | 0 | 0.36 | 0.64 | 1.83 | - | hinge |
| 6 | 180 | 0.354 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 0.543 | 0 | 0.09 | 0.91 | 0.541 | + | hinge |

## Groups

- negligible: units 1, 2, 6 (magnitude 0 together; a quarter of the tolerance is 0.0184)
- lines: units 0, 3 (each a constant or a line on the box; together one linear term)
- same line: units 4, 7 (gate angle 180 degrees, offset -0.537)
- the rest: units 5
