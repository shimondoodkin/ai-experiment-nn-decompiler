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
    return -0.45 * x0 + 1.607
```

Clean:

```python
def net(x0):
    return -0.45 * x0 + 1.607
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 24.2%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -13.2 +9.0 +21.7 +16.0 +9.9 +3.7 -2.5 -8.6 -14.8 -21.0
  - largest error per slice: +24.2 +20.0 +24.1 +19.1 +12.9 +6.8 +5.5 +11.7 +17.9 +24.1

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0315, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -2.27 | 0 | 0.00 | 1.00 | 0.255 | + | line |
| 1 | 180 | 0.277 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | -1.26 | 0 | 0.78 | 0.22 | 0.974 | + | hinge |
| 3 | 0 | -1.73 | 0 | 0.00 | 1.00 | 0.288 | + | line |
| 4 | 180 | -0.388 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 0 | 0.497 | 0 | 0.00 | 1.00 | 0.846 | - | line |
| 6 | 180 | 0.354 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 0.496 | 0 | 0.00 | 1.00 | 0.3 | - | line |

## Groups

- negligible: units 1, 4, 6 (magnitude 0 together; a quarter of the tolerance is 0.00788)
- lines: units 0, 3, 5, 7 (each a constant or a line on the box; together one linear term)
- the rest: units 2
