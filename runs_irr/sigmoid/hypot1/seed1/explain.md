# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (102 nodes):

```python
def net(x0, x1):
    u = 0.1016 * x0 * x0 + 0.1 * x1 * x1 + 0.0003326 * x0 * x1 - 0.002472 * x0 + 0.002026 * x1 + 0.6661
    return 2.39 * log(u) - 0.25 * relu(0.875 - u) - 0.35 * relu(u - 1.35) - 0.08 * relu(u - 1.25) + 2.071
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.769 * log(0.138 * x0 * x0 + 0.1371 * x1 * x1 - 0.0007288 * x0 * x1 - 0.003109 * x0 + 0.0065 * x1 + 0.5481) + 2.071
```

Clean:

```python
def net(x0, x1):
    return 1.769 * log(0.138 * x0 * x0 + 0.1371 * x1 * x1 - 0.0007288 * x0 * x1 - 0.003109 * x0 + 0.0065 * x1 + 0.5481) + 2.071
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.548, x1 = 0.428)
  - by distance from (0.0112, -0.0237), rings 0.285 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.1%, at the box edge (x0 = -2, x1 = 1.52)
  - by distance from (0.0112, -0.0237), rings 0.285 wide, mean: -1.1 -0.9 -0.6 -0.3 -0.1 -0.1 -0.2 -0.3 -0.7 -0.3; largest: +1.5 +1.6 +1.6 +1.4 +1.9 +2.1 +2.0 +2.0 +2.1 +2.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 +0.0 +0.6 +0.4 +0.5 +0.7 +0.2 -0.9 -1.1 -1.5 -1.4 -1.0; largest: +2.0 +2.0 +1.7 +2.0 +2.1 +1.9 +1.3 +1.9 +1.9 +2.1 +1.9 +1.9

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0386, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -16 | 1.86 | 2.2 | 0.41 | 0.36 | 0.23 | 0.00 | 0.00 | 0.336 | + | none (best 0.0247) |
| 1 | 107 | 1.88 | 1.98 | 0.47 | 0.32 | 0.20 | 0.00 | 0.00 | 1.34 | + | none (best 0.098) |
| 2 | 80 | 11.6 | 68.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0131 | - | constant |
| 3 | 106 | 0.494 | 1.97 | 0.12 | 0.32 | 0.38 | 0.18 | 0.00 | 1.04 | + | none (best 0.119) |
| 4 | -86 | -5.39 | 43.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0107 | + | constant |
| 5 | 10 | 1.32 | 2.32 | 0.25 | 0.37 | 0.38 | 0.00 | 0.00 | 1.2 | + | none (best 0.079) |
| 6 | -86 | -0.395 | 19.2 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0537 | + | line |
| 7 | -119 | 1.04 | 2.55 | 0.11 | 0.42 | 0.44 | 0.04 | 0.00 | 1.18 | + | none (best 0.0708) |
| 8 | -62 | -1.95 | 18.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00504 | - | constant |
| 9 | -31 | 0.869 | 1.88 | 0.21 | 0.33 | 0.36 | 0.10 | 0.00 | 0.61 | + | none (best 0.0706) |
| 10 | -85 | -3.1 | 33.6 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0166 | + | constant |
| 11 | -88 | 1.68 | 2.61 | 0.27 | 0.41 | 0.32 | 0.00 | 0.00 | 0.355 | + | none (best 0.0287) |
| 12 | 79 | 4.14 | 42 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0199 | - | line |
| 13 | -85 | -0.732 | 21 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0309 | + | line |
| 14 | -162 | 1.81 | 2.66 | 0.28 | 0.44 | 0.29 | 0.00 | 0.00 | 1.1 | + | none (best 0.0786) |
| 15 | -83 | 0.574 | 11.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0865 | + | line |

## Groups

- negligible: units 8 (magnitude 0.00504 together; a quarter of the tolerance is 0.00965)
- lines: units 2, 4, 6, 10, 12, 13, 15 (each a constant or a line on the box; together one linear term)
- the rest: units 0, 1, 3, 5, 7, 9, 11, 14
