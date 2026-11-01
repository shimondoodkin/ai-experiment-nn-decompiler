# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (28 nodes):

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.92
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.2487 * x0 * x0 + x1 * x1 - 0.001021 * x0 * x1 - 0.2412 * x0 - 0.0005735 * x1 < 0.9293
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.92
```

Share of the box where they decide differently: clean and faithful 0.18% (the dirt), faithful and the net 0.17% (what no conic follows), clean and the net 0.26%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0 | -0.25 | -0.2412 | 2.87% |
| x0² | 0.25 | 0.2487 | 1.27% |
| threshold | 0.92 | 0.9293 | 1.01% |
| x0·x1 | 0 | -0.001021 | 0.99% |
| x1 | 0 | -0.0005735 | 0.19% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (5.5, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -76 | 0.229 | 0 | 0.54 | 0.46 | 40 | - | hinge |
| 1 | 83 | 0.348 | 0 | 0.56 | 0.44 | 92.3 | - | hinge |
| 2 | 133 | 0.16 | 0 | 0.54 | 0.46 | 68.3 | - | hinge |
| 3 | 111 | 0.315 | 0 | 0.56 | 0.44 | 73.6 | - | hinge |
| 4 | 64 | -3.91 | 0 | 0.00 | 1.00 | 29.2 | + | line |
| 5 | 68 | 0.419 | 0 | 0.58 | 0.42 | 87.7 | - | hinge |
| 6 | 55 | -6.21 | 0 | 0.00 | 1.00 | 20.1 | + | line |
| 7 | -79 | 36.8 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | -88 | 0.313 | 0 | 0.55 | 0.45 | 19.3 | - | hinge |
| 9 | 19 | 0.595 | 0 | 0.61 | 0.39 | 35.5 | - | hinge |
| 10 | -78 | 0.194 | 0 | 0.53 | 0.47 | 18.7 | - | hinge |
| 11 | -47 | 0.946 | 0 | 0.70 | 0.30 | 67.5 | - | hinge |
| 12 | -106 | 0.315 | 0 | 0.55 | 0.45 | 113 | - | hinge |
| 13 | 62 | -5.36 | 0 | 0.00 | 1.00 | 19.8 | + | line |
| 14 | 175 | 0.351 | 0 | 0.56 | 0.44 | 39.4 | - | hinge |
| 15 | 105 | -3.01 | 0 | 0.02 | 0.98 | 1.38 | + | constant |

## Groups

- negligible: units 7 (magnitude 0 together; a quarter of the tolerance is 1.38)
- lines: units 4, 6, 13, 15 (each a constant or a line on the box; together one linear term)
- same line: units 0, 10 (gate angle -76 degrees, offset 0.229)
- the rest: units 1, 2, 3, 5, 8, 9, 11, 12, 14
