# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (7 nodes):

```python
def net(x0, x1):
    return x0 * x1
```

## Faithful fit

The quadratic formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no quadratic formula follows).

Faithful:

```python
def net(x0, x1):
    return -0.0009468 * x0 * x0 - 0.0005476 * x1 * x1 + 0.9965 * x0 * x1 + 0.00474 * x0 + 0.0006189 * x1 - 0.006792
```

Clean:

```python
def net(x0, x1):
    return x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -1.99, x1 = -1.99)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2; largest: +0.1 +0.1 +0.2 +0.2 +0.2 +0.3 +0.3 +0.4 +0.4 +0.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 +0.1 +0.1 +0.1 +0.1 +0.2 +0.2 +0.3 +0.2 +0.1 -0.0 +0.0; largest: +0.1 +0.2 +0.1 +0.1 +0.2 +0.3 +0.4 +0.5 +0.3 +0.1 +0.1 +0.1
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 1.99, x1 = -1.55)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 +0.0; largest: +0.1 +0.2 +0.2 +0.2 +0.3 +0.3 +0.4 +0.4 +0.4 +0.4
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 -0.1 -0.0 +0.0 +0.1 -0.0 -0.1 -0.1 -0.1 +0.1 +0.1 +0.1; largest: +0.2 +0.4 +0.3 +0.2 +0.3 +0.2 +0.3 +0.3 +0.3 +0.3 +0.4 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0·x1 | 1 | 0.9965 | 0.18% |
| x0 | 0 | 0.00474 | 0.12% |
| constant | 0 | -0.006792 | 0.09% |
| x0² | 0 | -0.0009468 | 0.05% |
| x1² | 0 | -0.0005476 | 0.03% |
| x1 | 0 | 0.0006189 | 0.02% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.158, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -46 | 2.94 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 1 | -11 | 3.14 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 2 | 30 | 2.16 | 0 | 0.98 | 0.02 | 0.138 | + | hinge |
| 3 | -65 | 3 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 4 | -55 | 2.19 | 0 | 0.98 | 0.02 | 0.219 | - | hinge |
| 5 | 27 | 6.06 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 6 | 130 | -1.49 | 0 | 0.11 | 0.89 | 0.488 | - | hinge |
| 7 | 49 | -1.22 | 0 | 0.16 | 0.84 | 0.807 | + | hinge |
| 8 | -46 | -0.448 | 0 | 0.35 | 0.65 | 0.876 | - | hinge |
| 9 | 134 | 0.712 | 0 | 0.72 | 0.28 | 0.555 | - | hinge |
| 10 | -148 | -1.25 | 0 | 0.16 | 0.84 | 0.158 | + | hinge |
| 11 | 131 | -1.29 | 0 | 0.15 | 0.85 | 0.426 | - | hinge |
| 12 | 110 | 3.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 13 | -136 | 1.96 | 0 | 0.95 | 0.05 | 0.283 | + | hinge |
| 14 | -137 | -0.57 | 0 | 0.32 | 0.68 | 0.799 | + | hinge |
| 15 | -134 | -0.277 | 0 | 0.41 | 0.59 | 0.841 | + | hinge |
| 16 | 48 | 1.77 | 0 | 0.93 | 0.07 | 0.363 | + | hinge |
| 17 | -22 | -2.05 | 0 | 0.03 | 0.97 | 0.23 | - | line |
| 18 | -47 | 1.06 | 0 | 0.80 | 0.20 | 0.505 | - | hinge |
| 19 | -123 | -1.35 | 0 | 0.13 | 0.87 | 0.336 | + | hinge |
| 20 | -130 | -1 | 0 | 0.21 | 0.79 | 0.774 | + | hinge |
| 21 | 46 | 1.39 | 0 | 0.87 | 0.13 | 0.393 | + | hinge |
| 22 | 170 | 2.4 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 23 | 59 | 2.16 | 0 | 0.98 | 0.02 | 0.15 | + | hinge |
| 24 | 36 | 6.47 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 25 | 52 | 0.658 | 0 | 0.70 | 0.30 | 0.288 | + | hinge |
| 26 | 46 | -0.279 | 0 | 0.41 | 0.59 | 0.881 | + | hinge |
| 27 | -170 | 3.45 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 28 | 132 | 2.34 | 0 | 0.99 | 0.01 | 0.155 | - | hinge |
| 29 | 103 | 2.46 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 30 | -61 | -1.58 | 0 | 0.10 | 0.90 | 0.296 | - | hinge |
| 31 | 51 | -0.633 | 0 | 0.30 | 0.70 | 0.533 | + | hinge |
| 32 | -25 | 1.54 | 0 | 0.90 | 0.10 | 0.134 | - | hinge |
| 33 | -47 | -0.969 | 0 | 0.21 | 0.79 | 0.886 | - | hinge |
| 34 | 134 | -0.252 | 0 | 0.42 | 0.58 | 0.645 | - | hinge |
| 35 | 136 | 1.55 | 0 | 0.90 | 0.10 | 0.361 | - | hinge |
| 36 | 157 | -1.02 | 0 | 0.22 | 0.78 | 0.128 | - | hinge |
| 37 | 143 | 3.01 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 38 | -52 | 1.77 | 0 | 0.93 | 0.07 | 0.326 | - | hinge |
| 39 | -132 | 0.971 | 0 | 0.78 | 0.22 | 0.481 | + | hinge |
| 40 | -16 | 1.99 | 0 | 0.97 | 0.03 | 0.0808 | - | hinge |
| 41 | 143 | 1.88 | 0 | 0.95 | 0.05 | 0.192 | - | hinge |
| 42 | -46 | 0.748 | 0 | 0.73 | 0.27 | 0.642 | - | hinge |
| 43 | 131 | -1.39 | 0 | 0.13 | 0.87 | 0.438 | - | hinge |
| 44 | -159 | -1.6 | 0 | 0.09 | 0.91 | 0.335 | + | hinge |
| 45 | -125 | 1.54 | 0 | 0.90 | 0.10 | 0.206 | + | hinge |
| 46 | -11 | 2.53 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 47 | 65 | -1.85 | 0 | 0.05 | 0.95 | 0.172 | + | line |
| 48 | -140 | 1.57 | 0 | 0.90 | 0.10 | 0.297 | + | hinge |
| 49 | -137 | 0.601 | 0 | 0.69 | 0.31 | 0.494 | + | hinge |
| 50 | 170 | 2.54 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 51 | 134 | -0.469 | 0 | 0.35 | 0.65 | 0.79 | - | hinge |
| 52 | 39 | 0.941 | 0 | 0.78 | 0.22 | 0.346 | + | hinge |
| 53 | 46 | 0.00107 | 0 | 0.50 | 0.50 | 0.804 | + | hinge |
| 54 | -134 | 3.26 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 55 | -35 | -1.16 | 0 | 0.17 | 0.83 | 0.238 | - | hinge |
| 56 | 175 | 2.83 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 57 | -135 | 2.35 | 0 | 0.99 | 0.01 | 0.172 | + | hinge |
| 58 | 133 | 1.23 | 0 | 0.84 | 0.16 | 0.409 | - | hinge |
| 59 | 104 | 2.43 | 0 | 1.00 | 0.00 | 0 | - | constant |
| 60 | 33 | -1.09 | 0 | 0.19 | 0.81 | 0.396 | + | hinge |
| 61 | 134 | -0.0354 | 0 | 0.49 | 0.51 | 0.637 | - | hinge |
| 62 | -46 | -0.191 | 0 | 0.43 | 0.57 | 0.746 | - | hinge |
| 63 | 123 | 1.96 | 0 | 0.96 | 0.04 | 0.166 | - | hinge |

## Groups

- negligible: units 0, 1, 3, 5, 12, 22, 24, 27, 29, 37, 46, 50, 54, 56, 59 (magnitude 0 together; a quarter of the tolerance is 0.0394)
- lines: units 17, 47 (each a constant or a line on the box; together one linear term)
- ring: 4 units (16, 21, 45, 48) with gates at distance 1.39-1.77, switch width ~0, magnitude ~0.33 each, positive
- ring: 4 units (2, 13, 23, 57) with gates at distance 1.96-2.35, switch width ~0, magnitude ~0.16 each, positive
- ring: 4 units (32, 38, 41, 63) with gates at distance 1.54-1.96, switch width ~0, magnitude ~0.18 each, negative
- same line: units 6, 11, 43 (gate angle 130 degrees, offset -1.49)
- same line: units 14, 25 (gate angle -137 degrees, offset -0.57)
- same line: units 31, 49 (gate angle 51 degrees, offset -0.633)
- the rest: units 4, 7, 8, 9, 10, 15, 18, 19, 20, 26, 28, 30, 33, 34, 35, 36, 39, 40, 42, 44, 51, 52, 53, 55, 58, 60, 61, 62
