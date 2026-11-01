# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (28 nodes):

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.2504 * x0 * x0 + x1 * x1 - 0.01295 * x0 * x1 - 0.2542 * x0 + 0.0005166 * x1 < 0.9392
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.24% (the dirt), faithful and the net 0.21% (what no conic follows), clean and the net 0.29%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | -0.01295 | 12.22% |
| x0 | -0.25 | -0.2542 | 1.33% |
| threshold | 0.95 | 0.9392 | 1.14% |
| x0² | 0.25 | 0.2504 | 0.38% |
| x1 | 0 | 0.0005166 | 0.16% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (4.82, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 39 | 1.58 | 0 | 0.80 | 0.20 | 56.8 | - | hinge |
| 1 | -66 | 1.12 | 0 | 0.70 | 0.30 | 49.1 | - | hinge |
| 2 | 166 | 5.67 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 95 | 2.57 | 0 | 0.93 | 0.07 | 0.00769 | + | constant |
| 4 | 59 | 5.66 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | -87 | 0.78 | 0 | 0.63 | 0.37 | 45.9 | - | hinge |
| 6 | -179 | 1.05 | 0 | 0.68 | 0.32 | 18.9 | - | hinge |
| 7 | -96 | 0.696 | 0 | 0.61 | 0.39 | 51.7 | - | hinge |
| 8 | 84 | 0.822 | 0 | 0.64 | 0.36 | 73.6 | - | hinge |
| 9 | -133 | 0.876 | 0 | 0.68 | 0.32 | 77.9 | - | hinge |
| 10 | 92 | 0.74 | 0 | 0.62 | 0.38 | 29.3 | - | hinge |
| 11 | 128 | 0.793 | 0 | 0.67 | 0.33 | 81.4 | - | hinge |
| 12 | 165 | 10.2 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -46 | 1.58 | 0 | 0.80 | 0.20 | 47.2 | - | hinge |
| 14 | -107 | 0.77 | 0 | 0.63 | 0.37 | 28.8 | - | hinge |
| 15 | 106 | 0.797 | 0 | 0.64 | 0.36 | 57.8 | - | hinge |

## Groups

- negligible: units 2, 3, 4, 12 (magnitude 0.00769 together; a quarter of the tolerance is 1.21)
- ring: 8 units (5, 7, 8, 9, 10, 11, 14, 15) with gates at distance 0.70-0.88, switch width ~0, magnitude ~55 each, negative
- the rest: units 0, 1, 6, 13
