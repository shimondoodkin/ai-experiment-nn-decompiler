# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (28 nodes):

```python
def net(x0):
    return 0.93 * log(x0 + 0.4346) + 0.25 * relu(1.25 - x0) + 0.6
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 0.9387 * log(x0 + 0.7308) + 0.5139
```

Clean:

```python
def net(x0):
    return 0.9387 * log(x0 + 0.7308) + 0.5139
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.5%, inside the box (x0 = 1.12)
  - mean error in 10 equal slices of x0: +0.9 -0.1 +2.3 +1.9 +1.4 +2.4 +2.4 +1.6 +0.3 -1.5
  - largest error per slice: +2.5 +2.5 +2.5 +2.5 +2.1 +2.5 +2.5 +2.1 +1.1 +2.5

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0369, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.432 | 0 | 0.00 | 1.00 | 0.865 | + | line |
| 1 | 180 | 0.277 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 0 | -9.22 | 0 | 0.00 | 1.00 | 0.0912 | + | line |
| 3 | 0 | -0.345 | 0 | 0.00 | 1.00 | 0.985 | + | line |
| 4 | 180 | -0.408 | 0 | 0.90 | 0.10 | 0.312 | - | hinge |
| 5 | 0 | 1.65 | 0 | 0.41 | 0.59 | 0.487 | - | hinge |
| 6 | 180 | 0.354 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 0.406 | 0 | 0.10 | 0.90 | 0.0804 | + | line |

## Groups

- negligible: units 1, 6 (magnitude 0 together; a quarter of the tolerance is 0.00923)
- lines: units 0, 2, 3, 7 (each a constant or a line on the box; together one linear term)
- the rest: units 4, 5
