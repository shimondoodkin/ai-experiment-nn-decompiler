# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (24 nodes):

```python
def net(x0):
    return -0.18 * x0 + 1.25 * relu(1.25 - x0) + 0.9
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return -0.4545 * x0 + 1.617
```

Clean:

```python
def net(x0):
    return -0.4545 * x0 + 1.617
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 24.2%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -13.0 +9.4 +21.7 +15.9 +9.7 +3.6 -2.5 -8.7 -14.8 -21.0
  - largest error per slice: +24.2 +20.5 +24.1 +18.9 +12.8 +6.7 +5.6 +11.7 +17.9 +24.0

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0319, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 11.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 180 | -1.25 | 0 | 0.78 | 0.22 | 0.98 | + | hinge |
| 2 | 180 | 0.367 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 180 | 0.267 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | 2.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 6.56 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 180 | 20.8 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 0.245 | 0 | 0.00 | 1.00 | 0.613 | - | line |

## Groups

- negligible: units 0, 2, 3, 4, 5, 6 (magnitude 0 together; a quarter of the tolerance is 0.00796)
- lines: units 7 (each a constant or a line on the box; together one linear term)
- the rest: units 1
