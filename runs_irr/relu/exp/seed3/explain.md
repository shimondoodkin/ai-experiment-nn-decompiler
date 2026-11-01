# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (35 nodes):

```python
def net(x0):
    return 3.5 * x0 + 3 * relu(1.3 - x0) + 1.75 * relu(x0 - 0.2) - 3
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return 1.79 * x0 + 2.146
```

Clean:

```python
def net(x0):
    return 1.79 * x0 + 2.146
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 19.8%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -16.0 -8.5 -1.0 +6.6 +14.1 +19.2 +18.2 +16.9 +8.1 -10.5
  - largest error per slice: +19.8 +12.2 +4.7 +10.4 +17.9 +19.8 +18.8 +17.5 +16.2 +19.7

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.143, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -2 | 0 | 0.00 | 1.00 | 5.71 | - | line |
| 1 | 0 | -4.7 | 0 | 0.00 | 1.00 | 1.35 | + | line |
| 2 | 180 | -1.23 | 0 | 0.19 | 0.81 | 9.96 | + | hinge |
| 3 | 0 | -4.96 | 0 | 0.00 | 1.00 | 0.919 | + | line |
| 4 | 180 | 12.3 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 180 | -2 | 0 | 0.00 | 1.00 | 6.13 | - | line |
| 6 | 0 | 0.097 | 0 | 0.53 | 0.47 | 3.01 | + | hinge |
| 7 | 0 | 2.6 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 4, 7 (magnitude 0 together; a quarter of the tolerance is 0.0358)
- lines: units 0, 1, 3, 5 (each a constant or a line on the box; together one linear term)
- the rest: units 2, 6
