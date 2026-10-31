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
    return 1.032 * exp(0.9847 * x0) - 0.03641
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.4%, at the box edge (x0 = 2)
  - mean error in 10 equal slices of x0: +0.4 +0.3 +0.3 +0.2 +0.1 +0.0 -0.1 -0.1 -0.1 +0.2
  - largest error per slice: +0.4 +0.3 +0.3 +0.2 +0.1 +0.1 +0.1 +0.1 +0.1 +0.4
- ripple (faithful minus net): largest 0.3%, at the box edge (x0 = 1.84)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.3 -0.1 +0.0 +0.1 +0.1 +0.2 +0.3 -0.2
  - largest error per slice: +0.3 +0.3 +0.3 +0.2 +0.1 +0.1 +0.1 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.032 | 3.28% |
| x0 inside exp | 1 | 0.9847 | 3.09% |
| offset | 0 | -0.03641 | 0.50% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -21 | 32.8 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.0587 | + | constant |
| 1 | 180 | 23.3 | 34 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0483 | - | constant |
| 2 | 0 | -14.5 | 26.5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.172 | + | rounded hinge |
| 3 | 0 | 0.741 | 2.46 | 0.07 | 0.38 | 0.46 | 0.08 | 0.00 | 2.19 | + | none (best 0.133) |
| 4 | 180 | -1.61 | 1.66 | 0.00 | 0.00 | 0.25 | 0.26 | 0.49 | 2.58 | - | none (best 0.203) |
| 5 | 180 | 24.2 | 32.8 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.074 | - | rounded hinge |
| 6 | 180 | 17.1 | 30.3 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0977 | - | rounded hinge |
| 7 | 0 | 2.08 | 1.07 | 0.75 | 0.17 | 0.08 | 0.00 | 0.00 | 2 | + | none (best 0.222) |

## Groups

- lines: units 0, 1 (each a constant or a line on the box; together one linear term)
- the rest: units 2, 3, 4, 5, 6, 7
