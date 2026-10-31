# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return log(x0)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.005 * log(x0 + 0.001736) + 0.005684
```

Clean:

```python
def net(x0):
    return log(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: -0.2 -0.2 -0.2 -0.2 -0.3 -0.3 -0.3 -0.3 -0.3 -0.4
  - largest error per slice: +0.3 +0.2 +0.2 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.4
- ripple (faithful minus net): largest 0.5%, at the box edge (x0 = 0.1)
  - mean error in 10 equal slices of x0: +0.0 +0.1 +0.2 +0.2 +0.2 +0.2 +0.3 +0.2 +0.3 +0.2
  - largest error per slice: +0.5 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.001736 | 0.44% |
| scale | 1 | 1.005 | 0.30% |
| offset | 0 | 0.005684 | 0.15% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0778, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -2.1 | 0 | 0.59 | 0.41 | 0.271 | - | hinge |
| 1 | 180 | 2.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 180 | 4.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 3 | 0 | 0.482 | 0 | 0.08 | 0.92 | 0.0222 | + | constant |
| 4 | 180 | -1.07 | 0 | 0.80 | 0.20 | 0.349 | - | hinge |
| 5 | 0 | 3.34 | 0 | 0.66 | 0.34 | 0.0041 | - | constant |
| 6 | 180 | -0.728 | 0 | 0.87 | 0.13 | 0.347 | - | hinge |
| 7 | 0 | 1.33 | 0 | 0.25 | 0.75 | 0.101 | - | hinge |
| 8 | 0 | -10.3 | 0 | 0.00 | 1.00 | 0.0417 | + | line |
| 9 | 180 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | 0 | -3 | 0 | 0.00 | 1.00 | 0.097 | + | line |
| 11 | 0 | 6.71 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 12 | 0 | 1.08 | 0 | 0.20 | 0.80 | 0.0758 | + | line |
| 13 | 180 | -0.297 | 0 | 0.96 | 0.04 | 0.327 | - | hinge |
| 14 | 180 | 7.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 180 | -2.81 | 0 | 0.45 | 0.55 | 0.282 | - | hinge |
| 16 | 0 | 14.9 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 17 | 180 | 27.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | 0 | 0.483 | 0 | 0.08 | 0.92 | 0.112 | + | line |
| 19 | 0 | 5.31 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 20 | 180 | -0.178 | 0 | 0.98 | 0.02 | 0.236 | - | hinge |
| 21 | 180 | 0.402 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 22 | 180 | -1.55 | 0 | 0.70 | 0.30 | 0.297 | - | hinge |
| 23 | 180 | -0.472 | 0 | 0.92 | 0.08 | 0.365 | - | hinge |
| 24 | 180 | 0.146 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 25 | 0 | 13.9 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 26 | 0 | -0.825 | 0 | 0.00 | 1.00 | 0.471 | + | line |
| 27 | 0 | 6.25 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 28 | 180 | 0.661 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 29 | 0 | -0.857 | 0 | 0.00 | 1.00 | 0.424 | + | line |
| 30 | 180 | -3.78 | 0 | 0.25 | 0.75 | 0.278 | - | hinge |
| 31 | 0 | 5.23 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 1, 2, 5, 9, 11, 14, 16, 17, 19, 21, 24, 25, 27, 28, 31 (magnitude 0.0041 together; a quarter of the tolerance is 0.0195)
- lines: units 3, 8, 10, 12, 18, 26, 29 (each a constant or a line on the box; together one linear term)
- same line: units 7, 22 (gate angle 0 degrees, offset 1.33)
- same line: units 13, 20, 23 (gate angle 180 degrees, offset -0.297)
- the rest: units 0, 4, 6, 15, 30
