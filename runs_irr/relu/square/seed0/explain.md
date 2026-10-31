# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

No formula passes the check. Near formula (square, 0 correction terms, 13 nodes), max error 2.6% of the output range (the check allows 2%):

```python
def net(x0):
    return 0.9959 * x0 * x0 - 0.02
```

Where it is off (formula minus net, % of the output range):
- largest error 2.6%, at the box edge (x0 = 2)
- mean error in 10 equal slices of x0: -0.9 -0.8 -0.6 -0.6 -0.6 -0.6 -0.4 -0.4 -0.3 -0.2
- largest error per slice: +2.3 +2.3 +2.1 +2.1 +2.2 +2.1 +2.2 +2.4 +2.4 +2.6

## Faithful fit

The square formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no square formula follows).

Faithful:

```python
def net(x0):
    return 0.9959 * (x0 - 0.003367) * (x0 - 0.003367) - 0.02535
```

Clean:

```python
def net(x0):
    return 0.9959 * (x0 - 0.003367) * (x0 - 0.003367) - 0.02535
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - largest error per slice: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.1%, inside the box (x0 = -1.2)
  - mean error in 10 equal slices of x0: -0.8 -0.7 -0.6 -0.6 -0.7 -0.8 -0.7 -0.7 -0.7 -0.7
  - largest error per slice: +2.1 +2.1 +2.1 +2.1 +2.1 +2.1 +2.1 +2.1 +2.1 +2.1

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0768, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 11.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 0 | 0.402 | 0 | 0.60 | 0.40 | 2.53 | + | hinge |
| 2 | 180 | 1.2 | 0 | 0.80 | 0.20 | 1.29 | + | hinge |
| 3 | 180 | 0.4 | 0 | 0.60 | 0.40 | 2.56 | + | hinge |
| 4 | 180 | 2.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 2.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 180 | 20.8 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 0 | 1.2 | 0 | 0.80 | 0.20 | 1.28 | + | hinge |

## Groups

- negligible: units 0, 4, 5, 6 (magnitude 0 together; a quarter of the tolerance is 0.0192)
- the rest: units 1, 2, 3, 7
