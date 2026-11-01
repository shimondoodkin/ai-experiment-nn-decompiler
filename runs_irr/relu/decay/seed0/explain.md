# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (14 nodes):

```python
def net(x0):
    return 3 * exp(-0.5 * x0)
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0):
    return 2.987 * exp(-0.5013 * x0) + 0.003295
```

Clean:

```python
def net(x0):
    return 3 * exp(-0.5 * x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.3%, at the box edge (x0 = -1)
  - mean error in 10 equal slices of x0: +0.3 +0.3 +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1
  - largest error per slice: +0.3 +0.3 +0.2 +0.2 +0.2 +0.2 +0.1 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 1.2%, inside the box (x0 = -0.562)
  - mean error in 10 equal slices of x0: -0.4 -0.8 +0.4 -1.0 -0.2 +0.2 -0.9 -0.7 +0.5 -0.4
  - largest error per slice: +1.2 +1.2 +1.2 +1.1 +0.9 +1.1 +1.0 +1.0 +1.0 +1.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 3 | 2.987 | 0.51% |
| x0 inside exp | -0.5 | -0.5013 | 0.15% |
| offset | 0 | 0.003295 | 0.08% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0833, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 11.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | 180 | -2.51 | 0 | 0.12 | 0.88 | 1.56 | + | hinge |
| 2 | 180 | 0.089 | 0 | 0.77 | 0.23 | 0.722 | + | hinge |
| 3 | 180 | -1.03 | 0 | 0.49 | 0.51 | 1.15 | + | hinge |
| 4 | 180 | 2.48 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | 0 | 5.02 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 180 | 20.8 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 7 | 180 | -3.09 | 0 | 0.00 | 1.00 | 0.736 | + | line |

## Groups

- negligible: units 0, 4, 5, 6 (magnitude 0 together; a quarter of the tolerance is 0.0208)
- lines: units 7 (each a constant or a line on the box; together one linear term)
- the rest: units 1, 2, 3
