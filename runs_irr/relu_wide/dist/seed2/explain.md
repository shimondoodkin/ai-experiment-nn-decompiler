# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

A family program passes the check (18 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(0.9963 * x0 * x0 + 0.9958 * x1 * x1 - 0.0002926 * x0 * x1 - 0.004683 * x0 + 0.007179 * x1 + 0.002327) - 0.003267
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1)
```

% of the output range:
- distortion (clean minus faithful): largest 0.9%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0, 0), rings 0.423 wide, mean: -0.0 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.3; largest: +0.9 +0.2 +0.2 +0.2 +0.3 +0.3 +0.3 +0.3 +0.3 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3 +0.2; largest: +0.5 +0.7 +0.4 +0.9 +0.6 +0.5 +0.5 +0.8 +0.3 +0.3 +0.4 +0.7
- ripple (faithful minus net): largest 0.2%, inside the box (x0 = -0.00204, x1 = 0.00898)
  - by distance from (0.00235, -0.0036), rings 0.424 wide, mean: +0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.2 +0.2 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.2 +0.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.0 -0.0 -0.0 +0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0; largest: +0.2 +0.2 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.2 +0.1

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0 | 0.002327 | 0.71% |
| x1² inside sqrt | 1 | 0.9958 | 0.15% |
| x0² inside sqrt | 1 | 0.9963 | 0.13% |
| x1 inside sqrt | 0 | 0.007179 | 0.09% |
| offset | 0 | -0.003267 | 0.08% |
| x0 inside sqrt | 0 | -0.004683 | 0.06% |
| x0·x1 inside sqrt | 0 | -0.0002926 | 0.01% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0839, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -38 | 4.49 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 0 | 3.06 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 2 | 2 | 4.54 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | -85 | 0.0635 | 0 | 0.51 | 0.49 | 0.079 | + | hinge |
| 4 | -67 | 0.0333 | 0 | 0.51 | 0.49 | 0.536 | + | hinge |
| 5 | 27 | 6.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 123 | 0.00165 | 0 | 0.50 | 0.50 | 0.638 | + | hinge |
| 7 | 74 | -0.00247 | 0 | 0.50 | 0.50 | 0.282 | + | hinge |
| 8 | -88 | 3.44 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 9 | 90 | 3.01 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 10 | 179 | 3.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 11 | 92 | 3.14 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 12 | 93 | 5.8 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -98 | 0.00912 | 0 | 0.50 | 0.50 | 0.463 | + | hinge |
| 14 | -137 | 0.00198 | 0 | 0.50 | 0.50 | 0.44 | + | hinge |
| 15 | -154 | 0.00962 | 0 | 0.50 | 0.50 | 0.416 | + | hinge |
| 16 | 91 | 0.0155 | 0 | 0.50 | 0.50 | 0.214 | + | hinge |
| 17 | 101 | 3.55 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 18 | -33 | 4.7 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 19 | -77 | 2.62 | 0 | 0.94 | 0.06 | 2.51e-05 | - | constant |
| 20 | -118 | 0.0534 | 0 | 0.51 | 0.49 | 0.0383 | + | constant |
| 21 | 39 | 4.24 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 22 | 92 | 3.09 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 9 | 0.0052 | 0 | 0.50 | 0.50 | 0.434 | + | hinge |
| 24 | 36 | 6.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 25 | 76 | 0.0146 | 0 | 0.50 | 0.50 | 0.16 | + | hinge |
| 26 | 35 | 0.0381 | 0 | 0.51 | 0.49 | 0.471 | + | hinge |
| 27 | 166 | 5.79 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 92 | 0.197 | 0 | 0.53 | 0.47 | 0.019 | + | constant |
| 29 | 89 | 0.0316 | 0 | 0.51 | 0.49 | 0.0797 | + | hinge |
| 30 | -66 | 0.234 | 0 | 0.54 | 0.46 | 0.00627 | + | constant |
| 31 | 67 | 0.00332 | 0 | 0.50 | 0.50 | 0.628 | + | hinge |
| 32 | -48 | 0.00358 | 0 | 0.50 | 0.50 | 0.616 | + | hinge |
| 33 | -89 | 0.112 | 0 | 0.52 | 0.48 | 0.0627 | + | hinge |
| 34 | 88 | 3.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 35 | 140 | 0.0446 | 0 | 0.51 | 0.49 | 0.506 | + | hinge |
| 36 | 159 | 0.00296 | 0 | 0.50 | 0.50 | 0.623 | + | hinge |
| 37 | 27 | 4.07 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 38 | -77 | 0.00324 | 0 | 0.50 | 0.50 | 0.599 | + | hinge |
| 39 | -122 | 0.00309 | 0 | 0.50 | 0.50 | 0.583 | + | hinge |
| 40 | -10 | 0.0773 | 0 | 0.51 | 0.49 | 0.34 | + | hinge |
| 41 | 170 | 0.0723 | 0 | 0.51 | 0.49 | 0.346 | + | hinge |
| 42 | -24 | 3.97 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 43 | 95 | 0.0308 | 0 | 0.51 | 0.49 | 0.084 | + | hinge |
| 44 | -151 | 0.026 | 0 | 0.50 | 0.50 | 0.0901 | + | hinge |
| 45 | -39 | 0.122 | 0 | 0.53 | 0.47 | 0.183 | + | hinge |
| 46 | -11 | 3.58 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 47 | 27 | 0.122 | 0 | 0.52 | 0.48 | 0.025 | + | constant |
| 48 | -125 | 2.65 | 0 | 0.93 | 0.07 | 0.000355 | - | constant |
| 49 | -162 | 0.0746 | 0 | 0.51 | 0.49 | 0.327 | + | hinge |
| 50 | 97 | 3.33 | 0 | 1.00 | 0.00 | 1.58e-23 | - | constant |
| 51 | -180 | 3.04 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 52 | 18 | 0.0684 | 0 | 0.51 | 0.49 | 0.342 | + | hinge |
| 53 | 50 | 0.00637 | 0 | 0.50 | 0.50 | 0.586 | + | hinge |
| 54 | -179 | 0.00208 | 0 | 0.50 | 0.50 | 0.525 | + | hinge |
| 55 | -30 | 0.00115 | 0 | 0.50 | 0.50 | 0.673 | + | hinge |
| 56 | 173 | 3.54 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 57 | -178 | 3.13 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 58 | 90 | 3.01 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 59 | 96 | 0.0249 | 0 | 0.51 | 0.49 | 0.116 | + | hinge |
| 60 | 35 | -0.0943 | 0 | 0.48 | 0.52 | 0.161 | + | hinge |
| 61 | 174 | 3.31 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 62 | -24 | 4.59 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 63 | 113 | 0.125 | 0 | 0.52 | 0.48 | 0.166 | + | hinge |

## Groups

- negligible: units 0, 1, 2, 5, 8, 9, 10, 11, 12, 17, 18, 19, 21, 22, 24, 27, 28, 30, 34, 37, 42, 46, 48, 50, 51, 56, 57, 58, 61, 62 (magnitude 0.0256 together; a quarter of the tolerance is 0.021)
- lines: units 20, 47 (each a constant or a line on the box; together one linear term)
- ring: 5 units (31, 32, 36, 38, 39) with gates at distance 0.00-0.00, switch width ~0, magnitude ~0.62 each, positive
- ring: 4 units (40, 41, 49, 52) with gates at distance 0.07-0.08, switch width ~0, magnitude ~0.34 each, positive
- same line: units 3, 7, 13, 16, 25, 29, 33, 43, 59 (gate angle -85 degrees, offset 0.0635)
- same line: units 4, 6, 63 (gate angle -67 degrees, offset 0.0333)
- same line: units 14, 15, 26, 44, 53, 60 (gate angle -137 degrees, offset 0.00198)
- same line: units 23, 54 (gate angle 9 degrees, offset 0.0052)
- same line: units 35, 45, 55 (gate angle 140 degrees, offset 0.0446)
