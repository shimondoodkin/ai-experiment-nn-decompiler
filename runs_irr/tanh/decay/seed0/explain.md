# Explain

Inputs: x0 in [-1, 3]; output: regression.

## Formula

A family program passes the check (12 nodes):

```python
def net(x0):
    return 18 / (x0 + 3.5) - 2.186
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 18.71 / (x0 + 3.606) - 2.208
```

Clean:

```python
def net(x0):
    return 18 / (x0 + 3.5) - 2.186
```

% of the output range:
- distortion (clean minus faithful): largest 1.0%, inside the box (x0 = 1.93)
  - mean error in 10 equal slices of x0: +0.5 -0.2 -0.5 -0.8 -0.9 -1.0 -1.0 -1.0 -1.0 -1.0
  - largest error per slice: +1.0 +0.4 +0.7 +0.8 +0.9 +1.0 +1.0 +1.0 +1.0 +1.0
- ripple (faithful minus net): largest 1.0%, at the box edge (x0 = -1)
  - mean error in 10 equal slices of x0: -0.3 -0.8 -0.4 +0.1 +0.6 +0.9 +0.9 +0.6 +0.1 -0.6
  - largest error per slice: +1.0 +0.9 +0.7 +0.3 +0.8 +1.0 +1.0 +0.8 +0.4 +0.9

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside reciprocal | 3.5 | 3.606 | 6.87% |
| scale | 18 | 18.71 | 6.66% |
| offset | -2.186 | -2.208 | 0.52% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0853, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.281 | 2.6 | 0.17 | 0.41 | 0.42 | 0.00 | 0.00 | 1.15 | + | none (best 0.081) |
| 1 | 180 | -10.3 | 20.3 | 0.00 | 0.00 | 0.09 | 0.91 | 0.00 | 0.215 | + | rounded hinge |
| 2 | 0 | 14.9 | 28 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.151 | - | rounded hinge |
| 3 | 0 | 11.1 | 21.4 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.23 | - | rounded hinge |
| 4 | 0 | 16 | 22.8 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0433 | - | rounded hinge |
| 5 | 0 | 3.87 | 14.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.13 | - | rounded hinge |
| 6 | 0 | 0.789 | 3.27 | 0.00 | 0.14 | 0.61 | 0.25 | 0.00 | 1.17 | - | none (best 0.0614) |
| 7 | 0 | -1.11 | 1.75 | 0.00 | 0.00 | 0.14 | 0.27 | 0.59 | 1.17 | - | none (best 0.132) |

## Groups

- the rest: units 0, 1, 2, 3, 4, 5, 6, 7
