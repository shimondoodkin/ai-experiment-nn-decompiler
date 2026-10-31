# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 2.8% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.8%, inside the box (x0 = 0.689, x1 = -1.04)
- by distance from (0, 0), rings 0.423 wide, mean: +0.1 +0.4 -0.3 +0.2 -0.5 +0.1 +0.4 +0.1 -0.1 -0.1; largest: +2.6 +2.3 +2.8 +2.7 +2.3 +2.4 +2.4 +2.5 +2.6 +2.4
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.0 -0.2 -0.2 -0.1 +0.2 +0.6 +0.7 +0.3 -0.2 -0.3 -0.2; largest: +2.3 +2.0 +2.2 +2.5 +2.6 +2.2 +2.3 +2.5 +2.0 +2.5 +2.8 +1.8

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.002 * exp(-1.018 * x0 * x0 - 0.9962 * x1 * x1 + 0.01639 * x0 * x1 - 0.01019 * x0 - 0.0004559 * x1) + 0.0005466
```

Clean:

```python
def net(x0, x1):
    return 1.002 * exp(-1.018 * x0 * x0 - 0.9962 * x1 * x1 + 0.01639 * x0 * x1 - 0.01019 * x0 - 0.0004559 * x1) + 0.0005466
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00501, -0.00027), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.5%, inside the box (x0 = -2.51, x1 = -1.9)
  - by distance from (-0.00501, -0.00027), rings 0.424 wide, mean: +0.2 +0.4 -0.4 +0.1 -0.5 +0.1 +0.4 +0.2 -0.0 -0.0; largest: +2.4 +2.5 +2.5 +2.4 +2.3 +2.4 +2.4 +2.5 +2.5 +2.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 -0.0 -0.1 -0.1 -0.0 +0.2 +0.6 +0.8 +0.5 -0.1 -0.4 -0.4; largest: +2.3 +2.1 +2.2 +2.3 +2.5 +2.3 +2.4 +2.5 +2.5 +2.4 +2.5 +2.2

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0208, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -57 | 1.21 | 0 | 0.74 | 0.26 | 0.486 | + | hinge |
| 1 | 8 | 4.81 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 2 | 12 | 1.17 | 0 | 0.70 | 0.30 | 0.33 | + | hinge |
| 3 | -96 | -0.156 | 0 | 0.47 | 0.53 | 0.318 | - | hinge |
| 4 | -68 | 0.183 | 0 | 0.53 | 0.47 | 0.0516 | - | hinge |
| 5 | 27 | 6.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 105 | -1.18 | 0 | 0.30 | 0.70 | 0.518 | + | hinge |
| 7 | 57 | -1.41 | 0 | 0.23 | 0.77 | 0.618 | + | hinge |
| 8 | -46 | -0.134 | 0 | 0.47 | 0.53 | 0.241 | - | hinge |
| 9 | 125 | 0.0641 | 0 | 0.51 | 0.49 | 0.372 | - | hinge |
| 10 | -153 | 4.25 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 11 | 95 | 0.163 | 0 | 0.53 | 0.47 | 0.307 | - | hinge |
| 12 | 93 | 5.8 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -104 | 1.2 | 0 | 0.71 | 0.29 | 0.213 | + | hinge |
| 14 | -135 | -1.15 | 0 | 0.26 | 0.74 | 0.501 | + | hinge |
| 15 | -131 | -0.0505 | 0 | 0.49 | 0.51 | 0.346 | - | hinge |
| 16 | 68 | 0.227 | 0 | 0.54 | 0.46 | 0.256 | - | hinge |
| 17 | -75 | -1.14 | 0 | 0.30 | 0.70 | 0.225 | + | hinge |
| 18 | -36 | 1.18 | 0 | 0.74 | 0.26 | 0.521 | + | hinge |
| 19 | -116 | -0.95 | 0 | 0.32 | 0.68 | 0.432 | + | hinge |
| 20 | -118 | -1.36 | 0 | 0.24 | 0.76 | 0.707 | + | hinge |
| 21 | 17 | -0.0396 | 0 | 0.49 | 0.51 | 0.643 | - | hinge |
| 22 | 67 | 0.051 | 0 | 0.51 | 0.49 | 0.289 | - | hinge |
| 23 | -1 | 3.66 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 24 | 36 | 6.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 25 | 56 | -0.0403 | 0 | 0.49 | 0.51 | 0.4 | - | hinge |
| 26 | 28 | 0.24 | 0 | 0.55 | 0.45 | 0.186 | - | hinge |
| 27 | 171 | 5.34 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 89 | 1.19 | 0 | 0.70 | 0.30 | 0.315 | + | hinge |
| 29 | 83 | -0.13 | 0 | 0.48 | 0.52 | 0.359 | - | hinge |
| 30 | -46 | -1.22 | 0 | 0.25 | 0.75 | 0.512 | + | hinge |
| 31 | 91 | -1.19 | 0 | 0.30 | 0.70 | 0.486 | + | hinge |
| 32 | -9 | -0.145 | 0 | 0.47 | 0.53 | 0.65 | - | hinge |
| 33 | -66 | 0.138 | 0 | 0.52 | 0.48 | 0.27 | - | hinge |
| 34 | 125 | -0.172 | 0 | 0.47 | 0.53 | 0.32 | - | hinge |
| 35 | 149 | 1.2 | 0 | 0.73 | 0.27 | 0.442 | + | hinge |
| 36 | -54 | 4.39 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 37 | -64 | 4.59 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 38 | -84 | 0.139 | 0 | 0.52 | 0.48 | 0.375 | - | hinge |
| 39 | -134 | 2.83 | 0 | 0.95 | 0.05 | 0.0324 | - | hinge |
| 40 | -12 | 1.02 | 0 | 0.67 | 0.33 | 0.241 | + | hinge |
| 41 | 172 | 1.2 | 0 | 0.70 | 0.30 | 0.364 | + | hinge |
| 42 | -39 | 0.031 | 0 | 0.51 | 0.49 | 0.223 | - | hinge |
| 43 | 69 | 4.37 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 44 | -161 | 1.55 | 0 | 0.77 | 0.23 | 0.115 | + | hinge |
| 45 | -112 | 0.207 | 0 | 0.54 | 0.46 | 0.32 | - | hinge |
| 46 | 30 | 1.24 | 0 | 0.74 | 0.26 | 0.366 | + | hinge |
| 47 | 35 | 0.13 | 0 | 0.53 | 0.47 | 0.508 | - | hinge |
| 48 | -145 | 0.224 | 0 | 0.54 | 0.46 | 0.562 | - | hinge |
| 49 | 142 | 6.89 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 50 | -142 | 7.68 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 51 | 169 | -0.202 | 0 | 0.47 | 0.53 | 0.529 | - | hinge |
| 52 | -13 | 1.39 | 0 | 0.74 | 0.26 | 0.183 | + | hinge |
| 53 | 50 | 0.346 | 0 | 0.58 | 0.42 | 0.0148 | - | hinge |
| 54 | 179 | -1.21 | 0 | 0.30 | 0.70 | 0.121 | + | hinge |
| 55 | -149 | 64.2 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 56 | -165 | 1.1 | 0 | 0.69 | 0.31 | 0.385 | + | hinge |
| 57 | -174 | -0.0401 | 0 | 0.49 | 0.51 | 0.431 | - | hinge |
| 58 | 113 | 0.132 | 0 | 0.53 | 0.47 | 0.432 | - | hinge |
| 59 | 56 | -1.01 | 0 | 0.30 | 0.70 | 0.588 | + | hinge |
| 60 | 37 | -1.18 | 0 | 0.26 | 0.74 | 0.585 | + | hinge |
| 61 | 148 | -0.192 | 0 | 0.46 | 0.54 | 0.472 | - | hinge |
| 62 | -31 | -0.124 | 0 | 0.47 | 0.53 | 0.594 | - | hinge |
| 63 | 119 | 1.22 | 0 | 0.74 | 0.26 | 0.462 | + | hinge |

## Groups

- negligible: units 1, 5, 10, 12, 23, 24, 27, 36, 37, 43, 49, 50, 55 (magnitude 0 together; a quarter of the tolerance is 0.00521)
- ring: 11 units (0, 2, 13, 18, 28, 35, 40, 41, 46, 56, 63) with gates at distance 1.02-1.24, switch width ~0, magnitude ~0.37 each, positive
- ring: 4 units (33, 38, 47, 58) with gates at distance 0.13-0.14, switch width ~0, magnitude ~0.4 each, negative
- same line: units 3, 29 (gate angle -96 degrees, offset -0.156)
- same line: units 8, 9, 34, 42, 61, 62 (gate angle -46 degrees, offset -0.134)
- same line: units 15, 25, 53 (gate angle -131 degrees, offset -0.0505)
- same line: units 16, 22, 45 (gate angle 68 degrees, offset 0.227)
- the rest: units 4, 6, 7, 11, 14, 17, 19, 20, 21, 26, 30, 31, 32, 39, 44, 48, 51, 52, 54, 57, 59, 60
