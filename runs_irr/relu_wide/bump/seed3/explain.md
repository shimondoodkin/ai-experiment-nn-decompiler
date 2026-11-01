# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 2 correction terms, 51 nodes), max error 2.5% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = -x0 * x0 - x1 * x1
    return exp(u) + 0.5 * relu(u + 0.05) + 0.00106 * relu(8 - u)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.5%, inside the box (x0 = -2.48, x1 = -2.05)
- by distance from (0, 0), rings 0.423 wide, mean: +0.2 +0.2 -0.8 -0.4 -0.9 -0.2 +0.1 -0.0 +0.1 +0.1; largest: +2.3 +2.3 +2.2 +2.1 +2.4 +2.0 +2.1 +2.5 +2.3 +2.2
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.7 -0.0 -0.2 -0.3 -0.5 -0.6 +0.1 +0.7 -0.1 -0.5 -0.5 -0.7; largest: +2.1 +2.1 +2.4 +2.3 +2.4 +2.1 +2.1 +2.5 +2.3 +2.1 +2.3 +2.2

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.004 * exp(-1.056 * x0 * x0 - 1.052 * x1 * x1 - 0.009597 * x0 * x1 + 0.001474 * x0 + 0.00947 * x1) + 0.02082
```

Clean:

```python
def net(x0, x1):
    return 1.004 * exp(-1.056 * x0 * x0 - 1.052 * x1 * x1 - 0.009597 * x0 * x1 + 0.001474 * x0 + 0.00947 * x1) + 0.02082
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.000677, 0.0045), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.8%, inside the box (x0 = -0.123, x1 = 0.438)
  - by distance from (0.000677, 0.0045), rings 0.424 wide, mean: +1.0 +0.1 -1.4 -0.6 -0.5 +0.3 +0.5 +0.2 -0.1 -0.3; largest: +2.7 +2.8 +2.7 +2.3 +2.1 +2.4 +2.5 +2.7 +2.4 +2.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.5 +0.1 +0.1 +0.0 -0.4 -0.4 +0.2 +0.7 +0.0 -0.3 -0.4 -0.4; largest: +2.6 +2.4 +2.5 +2.8 +2.5 +2.6 +2.5 +2.7 +2.6 +2.7 +2.5 +2.6

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0211, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -148 | -0.0361 | 0 | 0.49 | 0.51 | 0.494 | - | hinge |
| 1 | -112 | 0.02 | 0 | 0.50 | 0.50 | 0.486 | - | hinge |
| 2 | -103 | -1.19 | 0 | 0.29 | 0.71 | 0.552 | + | hinge |
| 3 | 35 | 1.37 | 0 | 0.77 | 0.23 | 0.264 | + | hinge |
| 4 | 82 | -0.268 | 0 | 0.46 | 0.54 | 0.121 | - | hinge |
| 5 | 116 | 0.25 | 0 | 0.55 | 0.45 | 0.0895 | - | hinge |
| 6 | -177 | 0.304 | 0 | 0.55 | 0.45 | 0.191 | - | hinge |
| 7 | -178 | -0.903 | 0 | 0.35 | 0.65 | 0.169 | + | hinge |
| 8 | 115 | -0.0516 | 0 | 0.49 | 0.51 | 0.463 | - | hinge |
| 9 | 24 | -0.876 | 0 | 0.34 | 0.66 | 0.0924 | + | hinge |
| 10 | -28 | 1.23 | 0 | 0.73 | 0.27 | 0.411 | + | hinge |
| 11 | -54 | 0.207 | 0 | 0.54 | 0.46 | 0.171 | - | hinge |
| 12 | -151 | 1.26 | 0 | 0.74 | 0.26 | 0.386 | + | hinge |
| 13 | 158 | -0.189 | 0 | 0.47 | 0.53 | 0.446 | - | hinge |
| 14 | -133 | 5.92 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 15 | 126 | 0.111 | 0 | 0.52 | 0.48 | 0.389 | - | hinge |
| 16 | -138 | 1.22 | 0 | 0.74 | 0.26 | 0.201 | + | hinge |
| 17 | -158 | 0.174 | 0 | 0.53 | 0.47 | 0.383 | - | hinge |
| 18 | -96 | -0.145 | 0 | 0.47 | 0.53 | 0.565 | - | hinge |
| 19 | 127 | 1.15 | 0 | 0.73 | 0.27 | 0.48 | + | hinge |
| 20 | -71 | -1.06 | 0 | 0.31 | 0.69 | 0.39 | + | hinge |
| 21 | -124 | -1.52 | 0 | 0.21 | 0.79 | 0.27 | + | hinge |
| 22 | -20 | -0.11 | 0 | 0.48 | 0.52 | 0.59 | - | hinge |
| 23 | 38 | 6.19 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 24 | 101 | 0.259 | 0 | 0.55 | 0.45 | 0.235 | - | hinge |
| 25 | -36 | -1.15 | 0 | 0.27 | 0.73 | 0.347 | + | hinge |
| 26 | 93 | -1.19 | 0 | 0.30 | 0.70 | 0.501 | + | hinge |
| 27 | 1 | 1.25 | 0 | 0.71 | 0.29 | 0.262 | + | hinge |
| 28 | 100 | -0.179 | 0 | 0.47 | 0.53 | 0.283 | - | hinge |
| 29 | -35 | -0.186 | 0 | 0.46 | 0.54 | 0.444 | - | hinge |
| 30 | -3 | 0.0868 | 0 | 0.52 | 0.48 | 0.368 | - | hinge |
| 31 | -175 | 1.18 | 0 | 0.70 | 0.30 | 0.338 | + | hinge |
| 32 | -123 | 1.39 | 0 | 0.77 | 0.23 | 0.355 | + | hinge |
| 33 | -78 | -0.00655 | 0 | 0.50 | 0.50 | 0.306 | - | hinge |
| 34 | 40 | 0.0461 | 0 | 0.51 | 0.49 | 0.332 | - | hinge |
| 35 | 72 | -1.21 | 0 | 0.29 | 0.71 | 0.0544 | + | hinge |
| 36 | 157 | 1.46 | 0 | 0.76 | 0.24 | 0.201 | + | hinge |
| 37 | 21 | 0.228 | 0 | 0.54 | 0.46 | 0.398 | - | hinge |
| 38 | 84 | -0.149 | 0 | 0.48 | 0.52 | 0.366 | - | hinge |
| 39 | 57 | 1.15 | 0 | 0.73 | 0.27 | 0.424 | + | hinge |
| 40 | -121 | 4.41 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 41 | 161 | 1.05 | 0 | 0.68 | 0.32 | 0.355 | + | hinge |
| 42 | -46 | 0.0848 | 0 | 0.52 | 0.48 | 0.354 | - | hinge |
| 43 | -75 | 3.86 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 44 | -148 | 0.311 | 0 | 0.56 | 0.44 | 0.079 | - | hinge |
| 45 | 165 | -1.16 | 0 | 0.30 | 0.70 | 0.27 | + | hinge |
| 46 | 145 | -0.0722 | 0 | 0.49 | 0.51 | 0.333 | - | hinge |
| 47 | 78 | -1.16 | 0 | 0.30 | 0.70 | 0.343 | + | hinge |
| 48 | 126 | 1.67 | 0 | 0.82 | 0.18 | 0.0837 | + | hinge |
| 49 | -145 | -0.989 | 0 | 0.30 | 0.70 | 0.454 | + | hinge |
| 50 | 49 | -0.242 | 0 | 0.45 | 0.55 | 0.104 | - | hinge |
| 51 | -47 | 1.21 | 0 | 0.75 | 0.25 | 0.396 | + | hinge |
| 52 | -87 | -1.19 | 0 | 0.30 | 0.70 | 0.397 | + | hinge |
| 53 | -69 | 3.99 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 54 | -70 | -1.36 | 0 | 0.26 | 0.74 | 0.24 | + | hinge |
| 55 | 59 | -0.976 | 0 | 0.31 | 0.69 | 0.446 | + | hinge |
| 56 | -129 | -0.0237 | 0 | 0.49 | 0.51 | 0.411 | - | hinge |
| 57 | -121 | -0.0208 | 0 | 0.49 | 0.51 | 0.472 | - | hinge |
| 58 | 9 | 5.12 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 59 | 112 | -0.963 | 0 | 0.33 | 0.67 | 0.41 | + | hinge |
| 60 | -64 | 1.37 | 0 | 0.75 | 0.25 | 0.312 | + | hinge |
| 61 | -173 | -0.0249 | 0 | 0.49 | 0.51 | 0.506 | - | hinge |
| 62 | -161 | -1.2 | 0 | 0.29 | 0.71 | 0.328 | + | hinge |
| 63 | 0 | 3.05 | 0 | 1.00 | 0.00 | 0 | - | constant |

## Groups

- negligible: units 14, 23, 40, 43, 53, 58, 63 (magnitude 0 together; a quarter of the tolerance is 0.00528)
- ring: 12 units (3, 10, 12, 16, 19, 27, 31, 32, 39, 41, 51, 60) with gates at distance 1.05-1.39, switch width ~0, magnitude ~0.36 each, positive
- same line: units 0, 1, 17, 34, 44, 50, 56, 57 (gate angle -148 degrees, offset -0.0361)
- same line: units 4, 18, 38 (gate angle 82 degrees, offset -0.268)
- same line: units 11, 15, 42 (gate angle -54 degrees, offset 0.207)
- same line: units 13, 22 (gate angle 158 degrees, offset -0.189)
- same line: units 24, 28, 33 (gate angle 101 degrees, offset 0.259)
- same line: units 29, 46 (gate angle -35 degrees, offset -0.186)
- same line: units 30, 61 (gate angle -3 degrees, offset 0.0868)
- same line: units 35, 47 (gate angle 72 degrees, offset -1.21)
- the rest: units 2, 5, 6, 7, 8, 9, 20, 21, 25, 26, 36, 37, 45, 48, 49, 52, 54, 55, 59, 62
