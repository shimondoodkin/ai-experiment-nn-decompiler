# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (35 nodes):

```python
def net(x0):
    return 3.5 * x0 + 3 * relu(1.32 - x0) + 1.75 * relu(x0 - 0.25) - 3
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return 1.759 * x0 + 2.146
```

Clean:

```python
def net(x0):
    return 1.759 * x0 + 2.146
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 20.0%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: -16.2 -8.6 -1.0 +6.7 +14.3 +19.4 +18.4 +17.2 +8.1 -10.6
  - largest error per slice: +20.0 +12.4 +4.8 +10.5 +18.1 +20.0 +19.0 +17.8 +16.5 +20.0

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.141, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 11.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 0.0959 | 0 | 0.52 | 0.48 | 3.77 | + | hinge |
| 2 | 180 | -0.0928 | 0 | 0.48 | 0.52 | 0.653 | - | hinge |
| 3 | 180 | -0.0903 | 0 | 0.48 | 0.52 | 0.218 | - | hinge |
| 4 | 180 | 2.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 2.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 180 | 20.8 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 1.22 | 0 | 0.81 | 0.19 | 2.39 | + | hinge |

## Groups

- negligible: units 0, 4, 5, 6 (magnitude 0 together; a quarter of the tolerance is 0.0352)
- same line: units 1, 2, 3 (gate angle 0 degrees, offset 0.0959)
- the rest: units 7
