# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (reciprocal, 0 correction terms, 39 nodes), max error 3.8% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return -1 / (0.01029 * x0 * x0 + 0.01064 * x1 * x1 - 0.001009 * x1 + 0.1559) + 7
```

Where it is off (formula minus net, % of the output range):
- largest error 3.8%, at the box edge (x0 = -0.0277, x1 = -3)
- by distance from (0, 0.0474), rings 0.427 wide, mean: +2.8 +1.9 +0.7 -0.2 -0.4 -0.4 -0.2 -0.0 +0.1 +0.7; largest: +3.1 +3.3 +3.6 +3.7 +3.6 +3.5 +3.6 +3.8 +3.5 +1.9
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -1.0 -1.5 -0.6 -0.5 -0.8 -0.7 +0.1 +0.7 +2.1 +2.2 +0.8 -0.1; largest: +3.0 +3.6 +3.5 +3.7 +3.6 +3.1 +3.1 +3.1 +3.8 +3.7 +3.0 +3.0

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return -1 / (0.01284 * x0 * x0 + 0.0135 * x1 * x1 + 0.000163 * x0 * x1 + 0.0001119 * x0 - 0.001309 * x1 + 0.1654) + 6.521
```

Clean:

```python
def net(x0, x1):
    return -1 / (0.01284 * x0 * x0 + 0.0135 * x1 * x1 + 0.000163 * x0 * x1 + 0.0001119 * x0 - 0.001309 * x1 + 0.1654) + 6.521
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00467, 0.0485), rings 0.427 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.5%, at the box edge (x0 = -0.0277, x1 = -3)
  - by distance from (-0.00467, 0.0485), rings 0.427 wide, mean: -0.2 -0.6 -1.1 -1.1 -0.7 -0.3 -0.1 -0.2 -0.6 -0.5; largest: +1.0 +2.7 +3.3 +3.3 +3.3 +3.3 +3.3 +3.5 +3.1 +1.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -1.6 -1.8 -0.8 -1.0 -1.7 -1.7 -0.7 +0.2 +1.9 +1.8 +0.1 -0.8; largest: +3.2 +3.3 +3.3 +3.3 +3.3 +3.1 +3.2 +2.8 +3.5 +3.5 +2.9 +3.3

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0715, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 26.7 | 48 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0845 | - | rounded hinge |
| 1 | 79 | 38.4 | 58.3 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0553 | - | rounded hinge |
| 2 | -157 | 0.745 | 1.66 | 0.33 | 0.19 | 0.23 | 0.16 | 0.09 | 0.98 | + | none (best 0.117) |
| 3 | 112 | 1.53 | 2.89 | 0.25 | 0.33 | 0.35 | 0.07 | 0.00 | 2.02 | + | none (best 0.18) |
| 4 | 54 | 5.34 | 16.7 | 0.00 | 0.31 | 0.69 | 0.00 | 0.00 | 0.287 | - | rounded hinge |
| 5 | -99 | -38.6 | 59 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0357 | + | constant |
| 6 | -106 | -26.2 | 48.9 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.108 | + | rounded hinge |
| 7 | 159 | -0.699 | 1.67 | 0.09 | 0.17 | 0.23 | 0.19 | 0.33 | 0.947 | - | none (best 0.113) |
| 8 | -108 | -28.4 | 50 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.122 | + | rounded hinge |
| 9 | 82 | 26.5 | 51 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.101 | - | rounded hinge |
| 10 | -145 | 2.6 | 4 | 0.23 | 0.49 | 0.28 | 0.00 | 0.00 | 3.47 | + | none (best 0.245) |
| 11 | 70 | 33.7 | 52.3 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0593 | - | rounded hinge |
| 12 | 59 | 9.73 | 26.2 | 0.00 | 0.48 | 0.52 | 0.00 | 0.00 | 0.146 | - | rounded hinge |
| 13 | -31 | 2.53 | 4.08 | 0.21 | 0.49 | 0.30 | 0.00 | 0.00 | 3.72 | + | none (best 0.251) |
| 14 | -107 | -1.66 | 3.24 | 0.00 | 0.04 | 0.38 | 0.35 | 0.23 | 2.8 | - | none (best 0.175) |
| 15 | -107 | -32.3 | 52.7 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.072 | + | rounded hinge |

## Groups

- lines: units 5 (each a constant or a line on the box; together one linear term)
- same line: units 6, 9 (gate angle -106 degrees, offset -26.2)
- the rest: units 0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15
