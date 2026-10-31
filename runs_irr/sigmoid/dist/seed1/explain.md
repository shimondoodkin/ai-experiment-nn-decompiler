# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (sqrt, 0 correction terms, 24 nodes), max error 2.6% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 0.2) - 0.0753
```

Where it is off (formula minus net, % of the output range):
- largest error 2.6%, inside the box (x0 = -2.19, x1 = 0.22)
- by distance from (0, 0), rings 0.423 wide, mean: +0.8 +1.7 +1.2 +0.2 -0.7 -1.0 -1.0 -0.8 -1.1 -1.4; largest: +1.7 +2.6 +2.5 +1.7 +2.6 +2.6 +2.3 +2.5 +2.4 +2.6
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 -0.8 -0.3 -0.5 -0.6 -0.5 -0.3 -0.8 -0.5 -0.3 -0.5 -0.3; largest: +2.2 +1.6 +2.5 +2.2 +2.6 +2.6 +2.6 +2.3 +1.7 +2.4 +1.3 +2.3

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.016 * x0 * x0 + 0.9935 * x1 * x1 - 0.001979 * x0 * x1 + 0.01893 * x0 + 0.004362 * x1 + 0.1072) - 0.04132
```

Clean:

```python
def net(x0, x1):
    return sqrt(1.016 * x0 * x0 + 0.9935 * x1 * x1 - 0.001979 * x0 * x1 + 0.01893 * x0 + 0.004362 * x1 + 0.1072) - 0.04132
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00932, -0.0022), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.2%, at the box edge (x0 = -3, x1 = 1.13)
  - by distance from (-0.00932, -0.0022), rings 0.424 wide, mean: -0.8 +1.0 +1.1 +0.3 -0.3 -0.5 -0.4 -0.2 -0.4 -0.5; largest: +2.1 +2.1 +2.1 +2.1 +2.1 +2.1 +1.9 +2.2 +1.8 +1.8
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.8 -0.2 -0.1 -0.4 -0.3 -0.1 +0.0 -0.6 -0.6 -0.1 +0.1 +0.5; largest: +2.1 +2.0 +2.0 +2.0 +2.0 +2.2 +2.1 +2.0 +2.1 +2.0 +2.0 +2.0

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0779, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -74 | 1.41 | 2.01 | 0.39 | 0.22 | 0.26 | 0.12 | 0.00 | 0.854 | + | none (best 0.103) |
| 1 | 84 | 2.77 | 3.16 | 0.44 | 0.33 | 0.23 | 0.00 | 0.00 | 1.35 | + | none (best 0.115) |
| 2 | 141 | 0.997 | 2 | 0.29 | 0.26 | 0.27 | 0.13 | 0.04 | 1.1 | + | none (best 0.131) |
| 3 | 70 | 0.815 | 1.95 | 0.30 | 0.21 | 0.26 | 0.18 | 0.05 | 1.21 | + | none (best 0.145) |
| 4 | -71 | -1.3 | 10.1 | 0.00 | 0.00 | 0.92 | 0.08 | 0.00 | 0.0499 | - | line |
| 5 | -1 | 2.4 | 2.18 | 0.54 | 0.23 | 0.24 | 0.00 | 0.00 | 1.26 | + | none (best 0.0934) |
| 6 | -174 | 0.141 | 9.15 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.073 | + | line |
| 7 | -87 | 2.75 | 2.93 | 0.47 | 0.31 | 0.22 | 0.00 | 0.00 | 1.06 | + | none (best 0.0944) |
| 8 | 7 | -1.02 | 2.95 | 0.00 | 0.14 | 0.37 | 0.31 | 0.18 | 0.302 | - | none (best 0.0263) |
| 9 | -1 | 0.729 | 1.66 | 0.34 | 0.18 | 0.21 | 0.17 | 0.10 | 1.11 | + | none (best 0.132) |
| 10 | -70 | -1.08 | 13 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0301 | - | constant |
| 11 | -76 | 0.473 | 1.15 | 0.38 | 0.12 | 0.15 | 0.12 | 0.22 | 0.505 | + | none (best 0.0602) |
| 12 | -144 | 0.89 | 1.8 | 0.31 | 0.23 | 0.26 | 0.14 | 0.06 | 0.903 | + | none (best 0.107) |
| 13 | -93 | -0.235 | 13.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0109 | - | constant |
| 14 | -178 | 2.62 | 3.42 | 0.37 | 0.36 | 0.28 | 0.00 | 0.00 | 1.16 | + | none (best 0.105) |
| 15 | -124 | 0.982 | 4.94 | 0.00 | 0.32 | 0.63 | 0.05 | 0.00 | 0.199 | + | rounded hinge |

## Groups

- negligible: units 13 (magnitude 0.0109 together; a quarter of the tolerance is 0.0195)
- lines: units 4, 6, 10 (each a constant or a line on the box; together one linear term)
- ring: 3 units (3, 9, 12) with gates at distance 0.73-0.89, switch width ~1.8, magnitude ~1.1 each, positive
- ring: 4 units (1, 5, 7, 14) with gates at distance 2.40-2.77, switch width ~3, magnitude ~1.2 each, positive
- the rest: units 0, 2, 8, 11, 15
