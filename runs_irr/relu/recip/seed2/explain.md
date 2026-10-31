# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (37 nodes):

```python
def net(x0):
    return -1.75 * x0 + 1.625 * relu(x0 - 0.875) + 0.5 * relu(1.75 - x0) + 2.16
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return -0.4803 * x0 + 1.754
```

Clean:

```python
def net(x0):
    return -0.4803 * x0 + 1.754
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 22.9%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -5.6 +17.7 +20.3 +22.0 +17.2 +9.9 +2.7 -4.6 -11.9 -19.2
  - largest error per slice: +22.8 +19.2 +21.3 +22.8 +20.9 +13.6 +6.3 +8.3 +15.5 +22.9

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0336, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -3.39 | 0 | 0.00 | 1.00 | 0.142 | + | line |
| 1 | 180 | 3.72 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 0 | 0.495 | 0 | 0.00 | 1.00 | 0.0102 | - | constant |
| 3 | 180 | -1.81 | 0 | 0.63 | 0.37 | 0.588 | + | hinge |
| 4 | 0 | 0.497 | 0 | 0.00 | 1.00 | 0.512 | - | line |
| 5 | 180 | -10.7 | 0 | 0.00 | 1.00 | 0.0762 | + | line |
| 6 | 180 | -0.0382 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 7 | 180 | -0.908 | 0 | 0.88 | 0.12 | 0.635 | + | hinge |

## Groups

- negligible: units 1, 6 (magnitude 0 together; a quarter of the tolerance is 0.0084)
- lines: units 0, 2, 4, 5 (each a constant or a line on the box; together one linear term)
- the rest: units 3, 7
