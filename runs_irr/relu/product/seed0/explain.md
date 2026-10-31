# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (10 nodes):

```python
def net(x0, x1):
    return 0.9763 * x0 * x1
```

## Faithful fit

The quadratic formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no quadratic formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.01485 * x0 * x0 + 0.004076 * x1 * x1 + 0.9773 * x0 * x1 - 0.006614 * x0 - 0.009121 * x1 - 0.05397
```

Clean:

```python
def net(x0, x1):
    return 0.9763 * x0 * x1
```

% of the output range:
- distortion (clean minus faithful): largest 0.8%, inside the box (x0 = 0.186, x1 = 1.1)
  - by distance from (0, 0), rings 0.282 wide, mean: +0.7 +0.7 +0.6 +0.6 +0.5 +0.4 +0.3 +0.2 -0.0 -0.2; largest: +0.7 +0.8 +0.8 +0.8 +0.8 +0.8 +0.8 +0.7 +0.6 +0.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.5 +0.5 +0.7 +0.7 +0.4 +0.2 +0.1 +0.0 +0.4 +0.4 +0.3 +0.4; largest: +0.7 +0.8 +0.8 +0.8 +0.7 +0.7 +0.7 +0.7 +0.7 +0.7 +0.7 +0.7
- ripple (faithful minus net): largest 1.7%, inside the box (x0 = 1.37, x1 = 1.73)
  - by distance from (0, 0), rings 0.282 wide, mean: -0.6 -0.8 -0.6 -0.5 -0.5 -0.5 -0.2 -0.1 -0.1 +0.1; largest: +1.2 +1.4 +1.4 +1.6 +1.5 +1.7 +1.6 +1.7 +1.6 +1.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.6 -0.8 -0.8 -0.5 +0.2 -0.2 -0.3 -0.6 -0.6 -0.4 +0.2 -0.3; largest: +1.7 +1.7 +1.6 +1.6 +1.6 +1.3 +1.4 +1.5 +1.6 +1.6 +1.7 +1.6

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| x0² | 0 | 0.01485 | 0.77% |
| constant | 0 | -0.05397 | 0.70% |
| x1 | 0 | -0.009121 | 0.24% |
| x1² | 0 | 0.004076 | 0.21% |
| x0 | 0 | -0.006614 | 0.17% |
| x0·x1 | 0.9763 | 0.9773 | 0.05% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.154, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 134 | 0.275 | 0 | 0.59 | 0.41 | 1.37 | - | hinge |
| 1 | -135 | -0.0581 | 0 | 0.48 | 0.52 | 2.28 | + | hinge |
| 2 | 162 | 4.92 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 3 | 46 | 1.73 | 0 | 0.93 | 0.07 | 0.938 | + | hinge |
| 4 | 81 | 4.46 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 5 | -36 | -1.58 | 0 | 0.10 | 0.90 | 3.23 | - | hinge |
| 6 | -135 | -0.869 | 0 | 0.24 | 0.76 | 3.11 | + | hinge |
| 7 | 134 | -0.236 | 0 | 0.42 | 0.58 | 1.71 | - | hinge |
| 8 | 58 | -0.837 | 0 | 0.26 | 0.74 | 1.64 | + | hinge |
| 9 | -145 | 0.828 | 0 | 0.74 | 0.26 | 1.1 | + | hinge |
| 10 | 49 | -1.77 | 0 | 0.07 | 0.93 | 3.93 | + | hinge |
| 11 | 136 | 0.856 | 0 | 0.76 | 0.24 | 1.26 | - | hinge |
| 12 | 147 | 3.75 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 13 | -46 | 0.952 | 0 | 0.78 | 0.22 | 1.56 | - | hinge |
| 14 | -72 | -1.84 | 0 | 0.05 | 0.95 | 1.62 | - | hinge |
| 15 | 129 | -1.68 | 0 | 0.08 | 0.92 | 3.05 | - | hinge |

## Groups

- negligible: units 2, 4, 12 (magnitude 0 together; a quarter of the tolerance is 0.0385)
- the rest: units 0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15
