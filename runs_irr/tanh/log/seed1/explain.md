# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0) + 0.02
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.034 * log(x0 + 0.01644) - 0.02852
```

Clean:

```python
def net(x0):
    return log(x0) + 0.02
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, inside the box (x0 = 0.485)
  - mean error in 10 equal slices of x0: +0.7 +0.9 +0.7 +0.5 +0.3 +0.2 +0.1 -0.0 -0.1 -0.2
  - largest error per slice: +1.0 +1.0 +0.8 +0.6 +0.4 +0.3 +0.1 +0.1 +0.2 +0.3
- ripple (faithful minus net): largest 0.7%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.2 +0.0 +0.2 +0.3 +0.4 +0.5 +0.5 +0.6
  - largest error per slice: +0.7 +0.4 +0.2 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.01644 | 3.97% |
| scale | 1 | 1.034 | 2.04% |
| offset | 0.02 | -0.02852 | 1.26% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0768, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.205 | 1.13 | 0.00 | 0.00 | 0.11 | 0.14 | 0.75 | 0.644 | + | none (best 0.0658) |
| 1 | 180 | -5.28 | 10.2 | 0.00 | 0.00 | 0.72 | 0.28 | 0.00 | 0.319 | - | rounded hinge |
| 2 | 0 | -30.7 | 81.9 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0341 | - | constant |
| 3 | 0 | 0.533 | 1.83 | 0.00 | 0.00 | 0.23 | 0.23 | 0.54 | 0.808 | + | none (best 0.0676) |
| 4 | 180 | -0.0445 | 0.442 | 0.92 | 0.06 | 0.02 | 0.00 | 0.00 | 0.9 | - | none (best 0.143) |
| 5 | 0 | 1.63 | 2.78 | 0.00 | 0.10 | 0.42 | 0.36 | 0.12 | 0.497 | + | none (best 0.0299) |
| 6 | 180 | -2.8 | 6.19 | 0.00 | 0.00 | 0.92 | 0.08 | 0.00 | 0.706 | - | none (best 0.0224) |
| 7 | 180 | 185 | 217 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00212 | + | constant |

## Groups

- negligible: units 7 (magnitude 0.00212 together; a quarter of the tolerance is 0.0192)
- lines: units 2 (each a constant or a line on the box; together one linear term)
- same line: units 0, 4 (gate angle 0 degrees, offset 0.205)
- the rest: units 1, 3, 5, 6
