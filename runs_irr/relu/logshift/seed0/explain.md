# Explain

Inputs: x0 in [-1.5, 4]; output: regression.

## Formula

A family program passes the check (32 nodes):

```python
def net(x0):
    return x0 - 0.78 * relu(x0 + 0.75) - 0.25 * relu(1 - x0) + 1.5
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return 0.4413 * x0 + 0.3852
```

Clean:

```python
def net(x0):
    return 0.4413 * x0 + 0.3852
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 13.4%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: +5.2 -8.9 -11.5 -12.4 -12.6 -8.4 -3.6 +1.3 +6.1 +11.0
  - largest error per slice: +13.3 +11.0 +12.0 +12.9 +13.3 +10.8 +6.0 +3.7 +8.5 +13.4

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0485, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 11.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | -3.56 | 0 | 0.00 | 1.00 | 0.512 | + | line |
| 2 | 180 | 0.688 | 0 | 0.85 | 0.15 | 0.657 | - | hinge |
| 3 | 180 | -0.964 | 0 | 0.55 | 0.45 | 0.626 | - | hinge |
| 4 | 180 | 2.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 5.01 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 180 | 20.8 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | -0.677 | 0 | 0.15 | 0.85 | 0.631 | + | hinge |

## Groups

- negligible: units 0, 4, 5, 6 (magnitude 0 together; a quarter of the tolerance is 0.0121)
- lines: units 1 (each a constant or a line on the box; together one linear term)
- same line: units 2, 7 (gate angle 180 degrees, offset 0.688)
- the rest: units 3
