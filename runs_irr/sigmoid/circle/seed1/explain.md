# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (18 nodes):

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.9874 * x0 * x0 + x1 * x1 + 0.01485 * x0 * x1 - 0.007252 * x0 - 0.0002445 * x1 < 3.943
```

Clean:

```python
def net(x0, x1):
    return x0 * x0 + x1 * x1 < 4
```

Share of the box where they decide differently: clean and faithful 0.29% (the dirt), faithful and the net 0.15% (what no conic follows), clean and the net 0.32%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 0 | 0.01485 | 3.35% |
| x0² | 1 | 0.9874 | 2.86% |
| threshold | 3.97 | 3.943 | 0.68% |
| x0 | -0.01 | -0.007252 | 0.21% |
| x1 | 0 | -0.0002445 | 0.02% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.88, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -174 | -90.9 | 95.2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0151 | - | constant |
| 1 | 112 | 1.74 | 0.872 | 0.66 | 0.10 | 0.11 | 0.07 | 0.07 | 11.6 | - | none (best 1.38) |
| 2 | -178 | 1.72 | 0.906 | 0.64 | 0.09 | 0.11 | 0.10 | 0.06 | 11.7 | - | none (best 1.39) |
| 3 | 75 | 1.74 | 0.892 | 0.65 | 0.10 | 0.11 | 0.08 | 0.06 | 11.5 | - | none (best 1.37) |
| 4 | -33 | -1.74 | 0.823 | 0.08 | 0.06 | 0.09 | 0.09 | 0.69 | 5.34 | + | none (best 0.634) |
| 5 | -41 | 1.78 | 0.897 | 0.69 | 0.10 | 0.09 | 0.06 | 0.07 | 11.3 | - | none (best 1.34) |
| 6 | -38 | -1.72 | 0.87 | 0.08 | 0.06 | 0.09 | 0.10 | 0.68 | 5.36 | + | none (best 0.638) |
| 7 | 41 | 1.72 | 0.941 | 0.67 | 0.10 | 0.10 | 0.06 | 0.07 | 10.5 | - | none (best 1.26) |
| 8 | -174 | -103 | 96.6 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00922 | - | constant |
| 9 | -8 | 1.75 | 0.908 | 0.64 | 0.10 | 0.11 | 0.09 | 0.05 | 11.2 | - | none (best 1.33) |
| 10 | -158 | -1.76 | 0.861 | 0.06 | 0.07 | 0.11 | 0.10 | 0.66 | 5.43 | + | none (best 0.647) |
| 11 | -92 | 1.74 | 0.872 | 0.64 | 0.09 | 0.11 | 0.09 | 0.06 | 11.5 | - | none (best 1.37) |
| 12 | -145 | 1.77 | 0.925 | 0.67 | 0.10 | 0.10 | 0.06 | 0.06 | 11.8 | - | none (best 1.41) |
| 13 | 63 | -1.69 | 0.834 | 0.08 | 0.06 | 0.10 | 0.10 | 0.66 | 5.63 | + | none (best 0.669) |
| 14 | -174 | -82.6 | 93.5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0191 | - | constant |
| 15 | 112 | -1.71 | 0.907 | 0.07 | 0.07 | 0.12 | 0.10 | 0.64 | 5.41 | + | none (best 0.643) |

## Groups

- negligible: units 0, 8, 14 (magnitude 0.0434 together; a quarter of the tolerance is 0.22)
- ring: 8 units (1, 2, 3, 5, 7, 9, 11, 12) with gates at distance 1.72-1.78, switch width ~0.9, magnitude ~11 each, negative
- same line: units 4, 6 (gate angle -33 degrees, offset -1.74)
- the rest: units 10, 13, 15
