# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 1 correction terms, 38 nodes), max error 2.6% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = -x0 * x0 - x1 * x1
    return exp(u) + 0.001141 * relu(8 - u)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.6%, inside the box (x0 = 0.523, x1 = 1.13)
- by distance from (0, 0), rings 0.423 wide, mean: +0.5 +0.7 -0.1 +0.4 -0.4 +0.2 +0.4 +0.4 +0.1 +0.0; largest: +2.5 +2.5 +2.6 +2.6 +1.8 +2.3 +2.4 +2.5 +2.1 +1.9
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 +0.5 +0.6 +0.6 +0.6 +0.1 -0.2 -0.2 -0.3 -0.2 +0.3 +0.3; largest: +2.4 +2.6 +2.6 +2.5 +2.4 +2.4 +2.4 +2.4 +2.3 +2.3 +2.5 +2.1

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.9902 * exp(-1.027 * x0 * x0 - 1.028 * x1 * x1 + 0.004146 * x0 * x1 - 0.002767 * x0 - 0.005624 * x1) + 0.01939
```

Clean:

```python
def net(x0, x1):
    return 0.9902 * exp(-1.027 * x0 * x0 - 1.028 * x1 * x1 + 0.004146 * x0 * x1 - 0.002767 * x0 - 0.005624 * x1) + 0.01939
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00135, -0.00274), rings 0.423 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.7%, at the box edge (x0 = -2.31, x1 = -3)
  - by distance from (-0.00135, -0.00274), rings 0.423 wide, mean: +0.4 +0.3 -0.4 +0.5 -0.1 +0.5 +0.6 +0.3 -0.3 -0.7; largest: +2.5 +2.4 +2.4 +2.3 +1.6 +2.6 +2.6 +2.5 +2.7 +2.6
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.3 +0.4 +0.7 +0.7 +0.5 +0.2 -0.1 -0.3 -0.2 -0.1 +0.3 +0.4; largest: +2.1 +2.3 +2.3 +2.5 +2.6 +2.3 +2.4 +2.7 +2.3 +2.6 +2.5 +2.1

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0208, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -47 | -0.111 | 0 | 0.47 | 0.53 | 0.197 | - | hinge |
| 1 | 117 | -1.21 | 0 | 0.27 | 0.73 | 0.588 | + | hinge |
| 2 | 136 | 1.11 | 0 | 0.73 | 0.27 | 0.42 | + | hinge |
| 3 | 105 | 0.0219 | 0 | 0.51 | 0.49 | 0.495 | - | hinge |
| 4 | -152 | 5.3 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 5 | 16 | -0.95 | 0 | 0.33 | 0.67 | 0.108 | + | hinge |
| 6 | -83 | -1.18 | 0 | 0.30 | 0.70 | 0.607 | + | hinge |
| 7 | 86 | 3.92 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 8 | 1 | 3.83 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 10 | -0.043 | 0 | 0.49 | 0.51 | 0.534 | - | hinge |
| 10 | -122 | 1.22 | 0 | 0.74 | 0.26 | 0.512 | + | hinge |
| 11 | -99 | 1.21 | 0 | 0.70 | 0.30 | 0.337 | + | hinge |
| 12 | -180 | 1.19 | 0 | 0.70 | 0.30 | 0.221 | + | hinge |
| 13 | 27 | 0.286 | 0 | 0.55 | 0.45 | 0.0746 | - | hinge |
| 14 | 118 | 0.227 | 0 | 0.54 | 0.46 | 0.467 | - | hinge |
| 15 | 74 | -0.124 | 0 | 0.48 | 0.52 | 0.33 | - | hinge |
| 16 | -60 | 0.234 | 0 | 0.54 | 0.46 | 0.357 | - | hinge |
| 17 | -142 | 0.276 | 0 | 0.56 | 0.44 | 0.0865 | - | hinge |
| 18 | -5 | 5.25 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 19 | -66 | -0.95 | 0 | 0.32 | 0.68 | 0.0792 | + | hinge |
| 20 | -22 | -0.131 | 0 | 0.47 | 0.53 | 0.532 | - | hinge |
| 21 | -16 | -1.19 | 0 | 0.30 | 0.70 | 0.437 | + | hinge |
| 22 | 118 | 1.24 | 0 | 0.74 | 0.26 | 0.333 | + | hinge |
| 23 | 143 | -1.44 | 0 | 0.22 | 0.78 | 0.53 | + | hinge |
| 24 | -162 | 1.25 | 0 | 0.72 | 0.28 | 0.263 | + | hinge |
| 25 | -155 | 0.033 | 0 | 0.50 | 0.50 | 0.631 | - | hinge |
| 26 | -17 | 1.18 | 0 | 0.71 | 0.29 | 0.257 | + | hinge |
| 27 | 48 | 1.2 | 0 | 0.74 | 0.26 | 0.327 | + | hinge |
| 28 | -91 | 0.131 | 0 | 0.52 | 0.48 | 0.394 | - | hinge |
| 29 | 31 | 1.27 | 0 | 0.75 | 0.25 | 0.395 | + | hinge |
| 30 | 120 | -0.0328 | 0 | 0.49 | 0.51 | 0.284 | - | hinge |
| 31 | 44 | 0.0998 | 0 | 0.52 | 0.48 | 0.251 | - | hinge |
| 32 | 150 | 1.23 | 0 | 0.74 | 0.26 | 0.322 | + | hinge |
| 33 | 10 | 3.6 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 34 | -143 | 0.0363 | 0 | 0.51 | 0.49 | 0.498 | - | hinge |
| 35 | -44 | -1.57 | 0 | 0.20 | 0.80 | 0.178 | + | hinge |
| 36 | 15 | 1.18 | 0 | 0.70 | 0.30 | 0.223 | + | hinge |
| 37 | 14 | -0.982 | 0 | 0.33 | 0.67 | 5.48e-10 | + | constant |
| 38 | 30 | 0.361 | 0 | 0.57 | 0.43 | 0.0452 | - | hinge |
| 39 | 142 | -0.00824 | 0 | 0.50 | 0.50 | 0.576 | - | hinge |
| 40 | -144 | 1.21 | 0 | 0.74 | 0.26 | 0.501 | + | hinge |
| 41 | -105 | 5.87 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 42 | -49 | 0.13 | 0 | 0.53 | 0.47 | 0.213 | - | hinge |
| 43 | 127 | -1.22 | 0 | 0.26 | 0.74 | 0.27 | + | hinge |
| 44 | 89 | 0.169 | 0 | 0.53 | 0.47 | 0.324 | - | hinge |
| 45 | -99 | -1.19 | 0 | 0.30 | 0.70 | 0.572 | + | hinge |
| 46 | -33 | 5.43 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 47 | -148 | -0.901 | 0 | 0.32 | 0.68 | 0.117 | + | hinge |
| 48 | -38 | 4.31 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 49 | -172 | -0.192 | 0 | 0.47 | 0.53 | 0.141 | - | hinge |
| 50 | 1 | 1.18 | 0 | 0.70 | 0.30 | 0.243 | + | hinge |
| 51 | 77 | 4.56 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 52 | -122 | -0.12 | 0 | 0.48 | 0.52 | 0.556 | - | hinge |
| 53 | -5 | 0.185 | 0 | 0.53 | 0.47 | 0.306 | - | hinge |
| 54 | 30 | 0.365 | 0 | 0.57 | 0.43 | 0.041 | - | hinge |
| 55 | 72 | 0.133 | 0 | 0.52 | 0.48 | 0.364 | - | hinge |
| 56 | -21 | 0.231 | 0 | 0.54 | 0.46 | 0.391 | - | hinge |
| 57 | 99 | -1.18 | 0 | 0.30 | 0.70 | 0.596 | + | hinge |
| 58 | 62 | 1.22 | 0 | 0.73 | 0.27 | 0.368 | + | hinge |
| 59 | -122 | 0.167 | 0 | 0.53 | 0.47 | 0.47 | - | hinge |
| 60 | 159 | 0.00189 | 0 | 0.50 | 0.50 | 0.0901 | - | hinge |
| 61 | -37 | 1.02 | 0 | 0.71 | 0.29 | 0.444 | + | hinge |
| 62 | -75 | 0.255 | 0 | 0.54 | 0.46 | 0.0719 | - | hinge |
| 63 | 175 | 0.152 | 0 | 0.53 | 0.47 | 0.443 | - | hinge |

## Groups

- negligible: units 4, 7, 8, 18, 33, 37, 41, 46, 48, 51 (magnitude 5.48e-10 together; a quarter of the tolerance is 0.00519)
- ring: 15 units (2, 10, 11, 12, 22, 24, 26, 27, 29, 32, 36, 40, 50, 58, 61) with gates at distance 1.02-1.27, switch width ~0, magnitude ~0.33 each, positive
- ring: 6 units (28, 42, 44, 55, 59, 63) with gates at distance 0.13-0.17, switch width ~0, magnitude ~0.38 each, negative
- same line: units 0, 39 (gate angle -47 degrees, offset -0.111)
- same line: units 3, 62 (gate angle 105 degrees, offset 0.0219)
- same line: units 9, 49 (gate angle 10 degrees, offset -0.043)
- same line: units 13, 38, 54 (gate angle 27 degrees, offset 0.286)
- same line: units 14, 16, 30 (gate angle 118 degrees, offset 0.227)
- same line: units 17, 31, 34 (gate angle -142 degrees, offset 0.276)
- same line: units 20, 56, 60 (gate angle -22 degrees, offset -0.131)
- the rest: units 1, 5, 6, 15, 19, 21, 23, 25, 35, 43, 45, 47, 52, 53, 57
