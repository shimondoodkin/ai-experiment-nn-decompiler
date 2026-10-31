# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 3 correction terms, 57 nodes), max error 3.1% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = -1.25 * x0 * x0 - 1.25 * x1 * x1
    return exp(u) - 0.125 * relu(u + 0.8) + 0.01668 * relu(u + 6)
```

Where it is off (formula minus net, % of the output range):
- largest error 3.1%, inside the box (x0 = -1.63, x1 = -2.45)
- by distance from (0, 0), rings 0.423 wide, mean: -0.4 +0.1 -0.4 +0.9 +0.5 +0.0 +0.4 +0.2 +0.1 +0.3; largest: +2.9 +1.5 +2.0 +3.0 +2.4 +2.4 +3.1 +3.0 +2.4 +1.8
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 -0.1 +0.1 +0.3 +0.4 +0.4 +0.5 +0.6 +0.5 +0.3 +0.2 -0.1; largest: +2.6 +2.7 +2.6 +2.7 +2.4 +2.5 +2.3 +3.1 +2.8 +2.9 +2.7 +2.6

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.008 * exp(-1.025 * x0 * x0 - 1.006 * x1 * x1 + 0.006053 * x0 * x1 + 0.00707 * x0 + 0.007769 * x1) - 0.002898
```

Clean:

```python
def net(x0, x1):
    return 1.008 * exp(-1.025 * x0 * x0 - 1.006 * x1 * x1 + 0.006053 * x0 * x1 + 0.00707 * x0 + 0.007769 * x1) - 0.002898
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.00346, 0.00387), rings 0.423 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.9%, inside the box (x0 = -1.63, x1 = -2.45)
  - by distance from (0.00346, 0.00387), rings 0.423 wide, mean: +0.6 +0.4 -0.8 -0.3 -0.6 -0.0 +0.2 -0.1 -0.2 +0.0; largest: +2.4 +2.4 +2.4 +2.0 +2.2 +2.3 +2.9 +2.7 +2.7 +1.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.5 -0.4 -0.2 -0.0 +0.0 -0.1 -0.1 +0.2 +0.0 -0.1 -0.2 -0.6; largest: +2.3 +2.3 +2.2 +2.2 +2.2 +2.3 +2.4 +2.9 +2.1 +2.4 +2.7 +2.7

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0212, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 116 | 0.256 | 0 | 0.55 | 0.45 | 0.363 | - | hinge |
| 1 | -163 | -0.00363 | 0 | 0.50 | 0.50 | 0.705 | - | hinge |
| 2 | 148 | 0.179 | 0 | 0.54 | 0.46 | 0.4 | - | hinge |
| 3 | 94 | 0.123 | 0 | 0.52 | 0.48 | 0.235 | - | hinge |
| 4 | 82 | 5.09 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | -94 | 5.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | -132 | 0.0373 | 0 | 0.51 | 0.49 | 0.383 | - | hinge |
| 7 | 178 | 3.34 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 8 | 95 | 3.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | -146 | 0.277 | 0 | 0.55 | 0.45 | 0.161 | - | hinge |
| 10 | 83 | 0.0303 | 0 | 0.51 | 0.49 | 0.274 | - | hinge |
| 11 | 128 | 0.325 | 0 | 0.57 | 0.43 | 0.0404 | - | hinge |
| 12 | -87 | 5.78 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 13 | -35 | -1.46 | 0 | 0.22 | 0.78 | 0.579 | + | hinge |
| 14 | -169 | -1.18 | 0 | 0.30 | 0.70 | 0.562 | + | hinge |
| 15 | 82 | 3.48 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 16 | 171 | 1.21 | 0 | 0.71 | 0.29 | 0.351 | + | hinge |
| 17 | -98 | 0.201 | 0 | 0.53 | 0.47 | 0.267 | - | hinge |
| 18 | 146 | -1.35 | 0 | 0.24 | 0.76 | 0.748 | + | hinge |
| 19 | 38 | 2.71 | 0 | 0.94 | 0.06 | 0.0405 | - | hinge |
| 20 | -61 | 1.36 | 0 | 0.76 | 0.24 | 0.416 | + | hinge |
| 21 | -68 | 0.0856 | 0 | 0.51 | 0.49 | 0.5 | - | hinge |
| 22 | -92 | 1.2 | 0 | 0.70 | 0.30 | 0.362 | + | hinge |
| 23 | 57 | 1.46 | 0 | 0.78 | 0.22 | 0.259 | + | hinge |
| 24 | -180 | 0.202 | 0 | 0.53 | 0.47 | 0.291 | - | hinge |
| 25 | 63 | -0.188 | 0 | 0.47 | 0.53 | 0.354 | - | hinge |
| 26 | 12 | -1.17 | 0 | 0.30 | 0.70 | 0.648 | + | hinge |
| 27 | -32 | 0.129 | 0 | 0.52 | 0.48 | 0.51 | - | hinge |
| 28 | -61 | 0.131 | 0 | 0.52 | 0.48 | 0.267 | - | hinge |
| 29 | -123 | 1.43 | 0 | 0.77 | 0.23 | 0.345 | + | hinge |
| 30 | 57 | 1.07 | 0 | 0.71 | 0.29 | 0.348 | + | hinge |
| 31 | -176 | 3.28 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 32 | -14 | 0.00421 | 0 | 0.50 | 0.50 | 0.723 | - | hinge |
| 33 | 87 | 1.06 | 0 | 0.68 | 0.32 | 0.22 | + | hinge |
| 34 | 82 | 3.83 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 35 | 180 | -0.118 | 0 | 0.48 | 0.52 | 0.466 | - | hinge |
| 36 | 85 | 0.34 | 0 | 0.56 | 0.44 | 0.171 | - | hinge |
| 37 | 61 | 0.278 | 0 | 0.55 | 0.45 | 0.253 | - | hinge |
| 38 | -87 | 0.173 | 0 | 0.53 | 0.47 | 0.3 | - | hinge |
| 39 | -74 | -1.1 | 0 | 0.31 | 0.69 | 0.491 | + | hinge |
| 40 | 145 | 1.05 | 0 | 0.71 | 0.29 | 0.429 | + | hinge |
| 41 | -51 | 0.237 | 0 | 0.55 | 0.45 | 0.23 | - | hinge |
| 42 | -97 | 4.05 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 43 | -96 | -0.164 | 0 | 0.47 | 0.53 | 0.198 | - | hinge |
| 44 | -58 | -1.22 | 0 | 0.26 | 0.74 | 0.742 | + | hinge |
| 45 | 33 | 0.0359 | 0 | 0.51 | 0.49 | 0.489 | - | hinge |
| 46 | 29 | 1.19 | 0 | 0.73 | 0.27 | 0.296 | + | hinge |
| 47 | -140 | -1.06 | 0 | 0.28 | 0.72 | 0.292 | + | hinge |
| 48 | 33 | 1.54 | 0 | 0.79 | 0.21 | 0.11 | + | hinge |
| 49 | -123 | 1.02 | 0 | 0.70 | 0.30 | 0.448 | + | hinge |
| 50 | 33 | -0.158 | 0 | 0.47 | 0.53 | 0.25 | - | hinge |
| 51 | 167 | 6.61 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 52 | 148 | -0.788 | 0 | 0.34 | 0.66 | 1.92e-06 | + | constant |
| 53 | -118 | -0.0299 | 0 | 0.49 | 0.51 | 0.505 | - | hinge |
| 54 | 174 | 3.92 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 55 | 129 | 0.0569 | 0 | 0.51 | 0.49 | 0.431 | - | hinge |
| 56 | 115 | -0.936 | 0 | 0.33 | 0.67 | 0.486 | + | hinge |
| 57 | -39 | 0.0572 | 0 | 0.51 | 0.49 | 0.2 | - | hinge |
| 58 | -149 | 1.23 | 0 | 0.74 | 0.26 | 0.449 | + | hinge |
| 59 | 87 | 1.4 | 0 | 0.74 | 0.26 | 0.153 | + | hinge |
| 60 | -9 | 1.23 | 0 | 0.71 | 0.29 | 0.354 | + | hinge |
| 61 | 146 | -0.942 | 0 | 0.31 | 0.69 | 0.492 | + | hinge |
| 62 | -146 | -0.253 | 0 | 0.45 | 0.55 | 0.321 | - | hinge |
| 63 | -114 | -0.958 | 0 | 0.32 | 0.68 | 0.192 | + | hinge |

## Groups

- negligible: units 4, 5, 7, 8, 12, 15, 31, 34, 42, 51, 52, 54 (magnitude 1.92e-06 together; a quarter of the tolerance is 0.00531)
- ring: 10 units (16, 20, 22, 30, 33, 40, 46, 49, 58, 60) with gates at distance 1.02-1.36, switch width ~0, magnitude ~0.36 each, positive
- ring: 4 units (23, 29, 48, 59) with gates at distance 1.40-1.54, switch width ~0, magnitude ~0.21 each, positive
- ring: 4 units (0, 9, 37, 41) with gates at distance 0.24-0.28, switch width ~0, magnitude ~0.24 each, negative
- same line: units 2, 27, 57 (gate angle 148 degrees, offset 0.179)
- same line: units 3, 10, 17, 36, 38, 43 (gate angle 94 degrees, offset 0.123)
- same line: units 11, 21, 28, 55 (gate angle 128 degrees, offset 0.325)
- same line: units 25, 53 (gate angle 63 degrees, offset -0.188)
- same line: units 45, 50, 62 (gate angle 33 degrees, offset 0.0359)
- the rest: units 1, 6, 13, 14, 18, 19, 24, 26, 32, 35, 39, 44, 47, 56, 61, 63
