# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return 0.4 * x0 + 0.5
```

## Faithful fit

The identity formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no identity formula follows).

Faithful:

```python
def net(x0):
    return 0.4016 * x0 + 0.5233
```

Clean:

```python
def net(x0):
    return 0.4 * x0 + 0.5
```

% of the output range:
- distortion (clean minus faithful): largest 1.8%, at the box edge (x0 = 4)
  - mean error in 10 equal slices of x0: -1.5 -1.5 -1.5 -1.6 -1.6 -1.7 -1.7 -1.7 -1.8 -1.8
  - largest error per slice: +1.5 +1.5 +1.6 +1.6 +1.6 +1.7 +1.7 +1.8 +1.8 +1.8
- ripple (faithful minus net): largest 0.0%, at the box edge (x0 = 3.99)
  - mean error in 10 equal slices of x0: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant | 0.5 | 0.5233 | 1.45% |
| x0 | 0.4 | 0.4016 | 0.40% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0321, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 11.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | -1.05 | 0 | 0.00 | 1.00 | 0.598 | + | line |
| 2 | 180 | 0.367 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 180 | 0.267 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | 180 | 2.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 4.33 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 180 | 20.8 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | -0.388 | 0 | 0.00 | 1.00 | 1.01 | + | line |

## Groups

- negligible: units 0, 2, 3, 4, 5, 6 (magnitude 0 together; a quarter of the tolerance is 0.00803)
- lines: units 1, 7 (each a constant or a line on the box; together one linear term)
