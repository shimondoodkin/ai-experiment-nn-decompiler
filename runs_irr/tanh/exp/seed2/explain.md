# Explain

Inputs: x0 in [-2, 2]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return exp(x0)
```

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0):
    return 1.039 * exp(0.9807 * x0) - 0.0395
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.6%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.4 +0.3 +0.2 +0.2 +0.1 -0.0 -0.1 -0.2 -0.1 +0.3
  - largest error per slice: +0.4 +0.3 +0.3 +0.2 +0.1 +0.1 +0.2 +0.2 +0.2 +0.6
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 1.82)
  - mean error in 10 equal slices of x0: -0.3 -0.4 -0.3 -0.1 +0.0 +0.1 +0.1 +0.3 +0.2 -0.2
  - largest error per slice: +0.4 +0.4 +0.4 +0.2 +0.1 +0.1 +0.1 +0.4 +0.4 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.039 | 4.00% |
| x0 inside exp | 1 | 0.9807 | 3.88% |
| offset | 0 | -0.0395 | 0.55% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.698 | 2.38 | 0.08 | 0.37 | 0.45 | 0.10 | 0.00 | 2.05 | + | none (best 0.122) |
| 1 | 180 | 26.5 | 25 | 0.89 | 0.11 | 0.00 | 0.00 | 0.00 | 0.0216 | - | constant |
| 2 | 0 | 2.12 | 1.48 | 0.66 | 0.23 | 0.11 | 0.00 | 0.00 | 1.23 | + | none (best 0.151) |
| 3 | 0 | -11.2 | 19.9 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.261 | + | rounded hinge |
| 4 | 180 | 15.1 | 22.9 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.16 | - | rounded hinge |
| 5 | 0 | -29.3 | 25.4 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.0126 | + | constant |
| 6 | 180 | -1.66 | 1.82 | 0.00 | 0.00 | 0.26 | 0.28 | 0.46 | 2.15 | - | none (best 0.19) |
| 7 | 180 | -2.1 | 1.18 | 0.00 | 0.00 | 0.08 | 0.18 | 0.73 | 1.33 | - | none (best 0.176) |

## Groups

- negligible: units 1, 5 (magnitude 0.0342 together; a quarter of the tolerance is 0.036)
- same line: units 2, 7 (gate angle 0 degrees, offset 2.12)
- the rest: units 0, 3, 4, 6
