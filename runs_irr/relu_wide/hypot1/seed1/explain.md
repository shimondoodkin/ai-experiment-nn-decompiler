# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (21 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(1.012 * x0 * x0 + 1.01 * x1 * x1 - 0.001528 * x0 * x1 + 0.0171 * x0 + 0.002426 * x1 + 1.057) - 0.02608
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1)
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, inside the box (x0 = -1.79, x1 = -0.824)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.1 -0.0 +0.0 +0.1 +0.2 +0.2 +0.2 +0.2 +0.2 +0.1; largest: +0.2 +0.2 +0.3 +0.4 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.1 +0.0 +0.2 +0.3 +0.4 +0.5 +0.4 +0.3 +0.1 -0.1 -0.2; largest: +0.2 +0.2 +0.2 +0.3 +0.4 +0.5 +0.5 +0.5 +0.5 +0.3 +0.2 +0.2
- ripple (faithful minus net): largest 0.6%, inside the box (x0 = 0.568, x1 = -0.197)
  - by distance from (-0.00845, -0.00121), rings 0.283 wide, mean: +0.1 +0.0 -0.1 -0.1 -0.2 -0.2 -0.1 -0.1 -0.1 -0.0; largest: +0.5 +0.5 +0.6 +0.4 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.2 -0.2 -0.2 -0.1 +0.0 -0.1 -0.0 -0.0 -0.2 -0.2 -0.2 -0.2; largest: +0.5 +0.4 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.5 +0.4 +0.5 +0.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 1 | 1.057 | 1.40% |
| offset | 0 | -0.02608 | 1.30% |
| x0² inside sqrt | 1 | 1.012 | 0.53% |
| x1² inside sqrt | 1 | 1.01 | 0.44% |
| x0 inside sqrt | 0 | 0.0171 | 0.38% |
| x1 inside sqrt | 0 | 0.002426 | 0.05% |
| x0·x1 inside sqrt | 0 | -0.001528 | 0.05% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.04, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -162 | 2.68 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 1 | 137 | -0.883 | 0 | 0.24 | 0.76 | 0.163 | + | hinge |
| 2 | 151 | 0.283 | 0 | 0.58 | 0.42 | 0.218 | + | hinge |
| 3 | 104 | 2.53 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 4 | -95 | -4.21 | 0 | 0.00 | 1.00 | 0.0581 | + | line |
| 5 | -25 | -6.11 | 0 | 0.00 | 1.00 | 0.0281 | + | line |
| 6 | -76 | -1.07 | 0 | 0.22 | 0.78 | 0.155 | + | hinge |
| 7 | -1 | 2.59 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | -99 | -1.82 | 0 | 0.04 | 0.96 | 1.68e-06 | + | constant |
| 9 | 1 | 2.06 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 10 | -106 | 0.908 | 0 | 0.74 | 0.26 | 0.0612 | + | hinge |
| 11 | -94 | 3.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 12 | 152 | 1.24 | 0 | 0.84 | 0.16 | 0.0744 | + | hinge |
| 13 | 60 | 0.119 | 0 | 0.53 | 0.47 | 0.207 | + | hinge |
| 14 | 144 | 0.726 | 0 | 0.72 | 0.28 | 0.151 | + | hinge |
| 15 | 90 | -0.202 | 0 | 0.45 | 0.55 | 0.224 | + | hinge |
| 16 | -74 | 0.248 | 0 | 0.56 | 0.44 | 0.161 | + | hinge |
| 17 | -154 | -0.0553 | 0 | 0.48 | 0.52 | 0.276 | + | hinge |
| 18 | -6 | -0.251 | 0 | 0.44 | 0.56 | 0.224 | + | hinge |
| 19 | -70 | -0.533 | 0 | 0.36 | 0.64 | 0.207 | + | hinge |
| 20 | 0 | 2.04 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 21 | 7 | -1.04 | 0 | 0.24 | 0.76 | 0.105 | + | hinge |
| 22 | 137 | 0.259 | 0 | 0.59 | 0.41 | 0.218 | + | hinge |
| 23 | 132 | -1.7 | 0 | 0.08 | 0.92 | 0.21 | + | hinge |
| 24 | -170 | -0.378 | 0 | 0.40 | 0.60 | 0.132 | + | hinge |
| 25 | 173 | 0.557 | 0 | 0.64 | 0.36 | 0.0891 | + | hinge |
| 26 | -13 | 2.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 27 | 52 | 0.525 | 0 | 0.67 | 0.33 | 0.207 | + | hinge |
| 28 | -46 | 1.25 | 0 | 0.84 | 0.16 | 0.0445 | + | hinge |
| 29 | 30 | 0.402 | 0 | 0.62 | 0.38 | 0.154 | + | hinge |
| 30 | 134 | 3.99 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 31 | 78 | 0.98 | 0 | 0.75 | 0.25 | 0.0644 | + | hinge |
| 32 | -139 | 1.08 | 0 | 0.81 | 0.19 | 0.0842 | + | hinge |
| 33 | -9 | 2.49 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 34 | -128 | -0.122 | 0 | 0.46 | 0.54 | 0.23 | + | hinge |
| 35 | -33 | -1.7 | 0 | 0.08 | 0.92 | 0.146 | + | hinge |
| 36 | 11 | 0.798 | 0 | 0.70 | 0.30 | 0.0808 | + | hinge |
| 37 | 25 | -0.267 | 0 | 0.43 | 0.57 | 0.274 | + | hinge |
| 38 | 66 | 1.62 | 0 | 0.91 | 0.09 | 0.0373 | + | hinge |
| 39 | 159 | 0.837 | 0 | 0.72 | 0.28 | 0.0591 | + | hinge |
| 40 | -129 | 0.59 | 0 | 0.68 | 0.32 | 0.151 | + | hinge |
| 41 | -128 | 7.12 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 42 | -43 | 0.545 | 0 | 0.67 | 0.33 | 0.177 | + | hinge |
| 43 | 133 | -0.135 | 0 | 0.45 | 0.55 | 0.215 | + | hinge |
| 44 | 95 | 1.65 | 0 | 0.92 | 0.08 | 0.000608 | + | constant |
| 45 | -92 | -0.165 | 0 | 0.46 | 0.54 | 0.233 | + | hinge |
| 46 | 72 | -8.35 | 0 | 0.00 | 1.00 | 0.0608 | + | line |
| 47 | 119 | 2.99 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 48 | -169 | 2.68 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 49 | -126 | 0.253 | 0 | 0.58 | 0.42 | 0.247 | + | hinge |
| 50 | -15 | 0.548 | 0 | 0.64 | 0.36 | 0.147 | + | hinge |
| 51 | 66 | -1.29 | 0 | 0.15 | 0.85 | 0.132 | + | hinge |
| 52 | -106 | 0.47 | 0 | 0.62 | 0.38 | 0.101 | + | hinge |
| 53 | -19 | 0.178 | 0 | 0.55 | 0.45 | 0.167 | + | hinge |
| 54 | 5 | 1.37 | 0 | 0.84 | 0.16 | 0.0246 | + | hinge |
| 55 | 88 | 0.506 | 0 | 0.63 | 0.37 | 0.126 | + | hinge |
| 56 | -21 | 0.968 | 0 | 0.76 | 0.24 | 0.0565 | + | hinge |
| 57 | 100 | -0.7 | 0 | 0.32 | 0.68 | 0.255 | + | hinge |
| 58 | 46 | 0.966 | 0 | 0.78 | 0.22 | 0.126 | + | hinge |
| 59 | -109 | 1.81 | 0 | 0.95 | 0.05 | 0.037 | + | hinge |
| 60 | 175 | -0.0817 | 0 | 0.48 | 0.52 | 0.178 | + | hinge |
| 61 | -37 | 0.184 | 0 | 0.56 | 0.44 | 0.196 | + | hinge |
| 62 | -70 | -0.123 | 0 | 0.47 | 0.53 | 0.206 | + | hinge |
| 63 | -152 | 0.595 | 0 | 0.67 | 0.33 | 0.122 | + | hinge |

## Groups

- negligible: units 0, 3, 7, 8, 9, 11, 20, 26, 30, 33, 41, 44, 47, 48 (magnitude 0.00061 together; a quarter of the tolerance is 0.01)
- lines: units 4, 5, 46 (each a constant or a line on the box; together one linear term)
- ring: 8 units (25, 27, 40, 42, 50, 52, 55, 63) with gates at distance 0.47-0.59, switch width ~0, magnitude ~0.14 each, positive
- ring: 5 units (10, 36, 39, 56, 58) with gates at distance 0.80-0.97, switch width ~0, magnitude ~0.061 each, positive
- ring: 4 units (12, 28, 31, 32) with gates at distance 0.98-1.25, switch width ~0, magnitude ~0.069 each, positive
- same line: units 13, 34 (gate angle 60 degrees, offset 0.119)
- same line: units 43, 61 (gate angle 133 degrees, offset -0.135)
- the rest: units 1, 2, 6, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 29, 35, 37, 38, 45, 49, 51, 53, 54, 57, 59, 60, 62
