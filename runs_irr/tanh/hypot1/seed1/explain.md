# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

No formula passes the check. Near formula (log, 4 correction terms, 87 nodes), max error 3.4% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = 0.01695 * x0 * x0 + 0.01813 * x1 * x1 + 1
    return 119 * log(u) - 97 * relu(u - 1) + 143 * relu(1 - u) - relu(1 - u) - 0.75 * relu(u + 0.375) + 2.076
```

Where it is off (formula minus net, % of the output range):
- largest error 3.3%, at the box edge (x0 = 1.9, x1 = -2)
- by distance from (0, 0), rings 0.282 wide, mean: -0.8 -0.8 -0.8 -0.8 -0.7 -0.5 -0.0 +0.3 +0.5 +0.3; largest: +1.0 +1.4 +2.0 +2.5 +2.6 +2.5 +2.7 +3.0 +3.1 +3.3
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -1.4 -0.7 +0.1 +1.2 +0.5 -0.9 -1.2 +0.1 +0.9 -0.1 -0.8 -1.5; largest: +3.0 +2.8 +2.7 +2.8 +3.3 +3.1 +2.6 +3.0 +1.9 +1.9 +3.3 +3.0

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.711 * log(0.1414 * x0 * x0 + 0.1456 * x1 * x1 + 0.001425 * x0 * x1 - 0.003094 * x0 + 0.001267 * x1 + 0.5357) + 2.076
```

Clean:

```python
def net(x0, x1):
    return 1.711 * log(0.1414 * x0 * x0 + 0.1456 * x1 * x1 + 0.001425 * x0 * x1 - 0.003094 * x0 + 0.001267 * x1 + 0.5357) + 2.076
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.548, x1 = 0.428)
  - by distance from (0.011, -0.0044), rings 0.283 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.3%, at the box edge (x0 = 2, x1 = -0.874)
  - by distance from (0.011, -0.0044), rings 0.283 wide, mean: -2.5 -1.9 -0.9 -0.1 +0.4 +0.5 +0.4 -0.0 -0.2 -0.1; largest: +2.6 +2.4 +1.7 +1.6 +2.5 +2.6 +2.5 +3.3 +3.2 +3.1
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.7 -0.6 +0.2 +1.3 +0.9 +0.4 +0.2 +0.6 +0.9 -0.5 -1.4 -1.1; largest: +2.9 +2.9 +2.6 +2.6 +3.1 +3.2 +2.7 +3.0 +2.6 +2.6 +3.1 +3.3

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0391, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -58 | 1.46 | 3.34 | 0.05 | 0.51 | 0.44 | 0.00 | 0.00 | 3.11 | + | none (best 0.208) |
| 1 | 146 | 60.6 | 101 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.015 | - | constant |
| 2 | 148 | 51.5 | 95 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0287 | - | rounded hinge |
| 3 | 147 | 34.1 | 82.4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0438 | - | rounded hinge |
| 4 | -33 | -51 | 95.5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0234 | + | rounded hinge |
| 5 | 151 | 21.3 | 68 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0686 | - | rounded hinge |
| 6 | -31 | -27.7 | 75.8 | 0.00 | 0.00 | 0.72 | 0.28 | 0.00 | 0.0663 | + | rounded hinge |
| 7 | 178 | 1.09 | 17.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0787 | - | line |
| 8 | -33 | -64.3 | 103 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0117 | + | constant |
| 9 | 149 | 29.8 | 78.1 | 0.00 | 0.66 | 0.34 | 0.00 | 0.00 | 0.0524 | - | rounded hinge |
| 10 | -34 | -45 | 91.2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.031 | + | rounded hinge |
| 11 | 63 | 1.47 | 3.48 | 0.04 | 0.51 | 0.45 | 0.00 | 0.00 | 3.28 | + | none (best 0.22) |
| 12 | 148 | 33.6 | 81.9 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0446 | - | rounded hinge |
| 13 | -32 | -34.7 | 83 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0454 | + | rounded hinge |
| 14 | -178 | 1.36 | 3.48 | 0.00 | 0.51 | 0.49 | 0.00 | 0.00 | 2.85 | + | none (best 0.201) |
| 15 | -29 | -20.4 | 66.7 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0792 | + | rounded hinge |

## Groups

- lines: units 1, 7, 8 (each a constant or a line on the box; together one linear term)
- ring: 3 units (0, 11, 14) with gates at distance 1.36-1.47, switch width ~3.5, magnitude ~3.1 each, positive
- the rest: units 2, 3, 4, 5, 6, 9, 10, 12, 13, 15
