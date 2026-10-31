# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (28 nodes):

```python
def net(x0):
    return 0.97 * log(x0 + 0.5) + 0.25 * relu(1.25 - x0) + 0.54
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 0.9476 * log(x0 + 0.7445) + 0.5014
```

Clean:

```python
def net(x0):
    return 0.9476 * log(x0 + 0.7445) + 0.5014
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.6%, inside the box (x0 = 0.416)
  - mean error in 10 equal slices of x0: +0.9 -0.2 +2.3 +2.1 +1.4 +2.3 +2.3 +1.6 +0.3 -1.5
  - largest error per slice: +2.5 +2.6 +2.6 +2.6 +1.9 +2.4 +2.4 +2.0 +1.0 +2.5

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.037, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.559 | 0 | 0.00 | 1.00 | 0.951 | + | line |
| 1 | 180 | 3.72 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 0 | -0.0138 | 0 | 0.00 | 1.00 | 0.0513 | + | line |
| 3 | 0 | -0.648 | 0 | 0.00 | 1.00 | 0.624 | + | line |
| 4 | 0 | 1.68 | 0 | 0.42 | 0.58 | 0.468 | - | hinge |
| 5 | 0 | -1.36 | 0 | 0.00 | 1.00 | 0.384 | + | line |
| 6 | 180 | 0.162 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 7 | 180 | -0.416 | 0 | 0.89 | 0.11 | 0.306 | - | hinge |

## Groups

- negligible: units 1, 6 (magnitude 0 together; a quarter of the tolerance is 0.00924)
- lines: units 0, 2, 3, 5 (each a constant or a line on the box; together one linear term)
- the rest: units 4, 7
