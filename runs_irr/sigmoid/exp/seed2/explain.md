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
    return 1.046 * exp(0.9784 * x0) - 0.046
```

Clean:

```python
def net(x0):
    return exp(x0)
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = -2)
  - mean error in 10 equal slices of x0: +0.4 +0.4 +0.3 +0.2 +0.1 -0.1 -0.2 -0.2 -0.2 +0.1
  - largest error per slice: +0.5 +0.4 +0.3 +0.2 +0.1 +0.1 +0.2 +0.3 +0.2 +0.5
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 1.82)
  - mean error in 10 equal slices of x0: -0.4 -0.4 -0.3 -0.1 +0.0 +0.1 +0.1 +0.3 +0.3 -0.2
  - largest error per slice: +0.4 +0.4 +0.4 +0.2 +0.1 +0.1 +0.2 +0.4 +0.4 +0.4

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.046 | 4.72% |
| x0 inside exp | 1 | 0.9784 | 4.34% |
| offset | 0 | -0.046 | 0.64% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.144, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.84 | 2.27 | 0.39 | 0.36 | 0.25 | 0.00 | 0.00 | 1.03 | + | none (best 0.1) |
| 1 | 180 | -1.83 | 2.22 | 0.00 | 0.00 | 0.25 | 0.35 | 0.40 | 0.856 | - | none (best 0.0808) |
| 2 | 0 | 2.06 | 1.13 | 0.73 | 0.18 | 0.09 | 0.00 | 0.00 | 1.01 | + | none (best 0.129) |
| 3 | 0 | 0.872 | 2.55 | 0.08 | 0.40 | 0.48 | 0.04 | 0.00 | 1.71 | + | none (best 0.109) |
| 4 | 0 | 0.547 | 4.02 | 0.00 | 0.26 | 0.74 | 0.00 | 0.00 | 0.395 | + | rounded hinge |
| 5 | 0 | 1.92 | 2.1 | 0.46 | 0.33 | 0.21 | 0.00 | 0.00 | 0.825 | + | none (best 0.0803) |
| 6 | 0 | 2.08 | 1.35 | 0.68 | 0.21 | 0.11 | 0.00 | 0.00 | 1.09 | + | none (best 0.124) |
| 7 | 0 | -3.61 | 15 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.289 | + | rounded hinge |

## Groups

- same line: units 0, 1, 2, 5, 6 (gate angle 0 degrees, offset 1.84)
- the rest: units 3, 4, 7
