# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (40 nodes):

```python
def net(x0):
    return -6 / (x0 + 1.25) + 0.2 * relu(1.5 - x0) - 2 * relu(0.4 - x0) + 2.56
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return -3.831 / (x0 + 0.8214) + 2.168
```

Clean:

```python
def net(x0):
    return -3.831 / (x0 + 0.8214) + 2.168
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0.1)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.9%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: +0.0 +1.2 +2.5 +3.7 +2.1 +1.7 +1.4 +0.5 -1.0 -2.9
  - largest error per slice: +3.8 +1.7 +3.7 +3.8 +3.2 +1.7 +1.7 +1.0 +1.9 +3.9

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0758, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.098 | 0 | 0.00 | 1.00 | 1.9 | + | line |
| 1 | 180 | 3.72 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 0 | 0.0965 | 0 | 0.00 | 1.00 | 0.156 | + | line |
| 3 | 180 | 0.369 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | 0 | 2.45 | 0 | 0.48 | 0.52 | 0.807 | - | hinge |
| 5 | 0 | 0.1 | 0 | 0.00 | 1.00 | 0.855 | + | line |
| 6 | 180 | -1.06 | 0 | 0.80 | 0.20 | 0.848 | - | hinge |
| 7 | 180 | -0.373 | 0 | 0.94 | 0.06 | 0.839 | - | hinge |

## Groups

- negligible: units 1, 3 (magnitude 0 together; a quarter of the tolerance is 0.019)
- lines: units 0, 2, 5 (each a constant or a line on the box; together one linear term)
- the rest: units 4, 6, 7
