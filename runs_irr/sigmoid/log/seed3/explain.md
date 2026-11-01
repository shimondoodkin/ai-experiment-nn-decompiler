# Explain

Inputs: x0 in [0.1, 5]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0) + 0.05
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.034 * log(x0 + 0.01848) - 0.03069
```

Clean:

```python
def net(x0):
    return log(x0) + 0.05
```

% of the output range:
- distortion (clean minus faithful): largest 1.8%, inside the box (x0 = 0.543)
  - mean error in 10 equal slices of x0: +1.4 +1.7 +1.5 +1.3 +1.2 +1.0 +0.9 +0.8 +0.7 +0.6
  - largest error per slice: +1.8 +1.8 +1.6 +1.4 +1.2 +1.1 +1.0 +0.9 +0.8 +0.7
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: -0.2 -0.4 -0.2 -0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7
  - largest error per slice: +0.8 +0.7 +0.2 +0.2 +0.3 +0.4 +0.5 +0.5 +0.6 +0.9

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.01848 | 4.45% |
| offset | 0.05 | -0.03069 | 2.12% |
| scale | 1 | 1.034 | 2.06% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0762, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | 0.0678 | 0.737 | 0.88 | 0.10 | 0.02 | 0.00 | 0.00 | 0.557 | - | none (best 0.0955) |
| 1 | 180 | -2.6 | 6.64 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.601 | - | rounded hinge |
| 2 | 0 | 0.987 | 11.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.213 | + | rounded hinge |
| 3 | 180 | -0.451 | 1.75 | 0.57 | 0.22 | 0.21 | 0.00 | 0.00 | 0.805 | - | none (best 0.0738) |
| 4 | 0 | 2.17 | 7.45 | 0.00 | 0.00 | 0.99 | 0.01 | 0.00 | 0.428 | + | rounded hinge |
| 5 | 180 | 0.0269 | 0.572 | 0.91 | 0.07 | 0.02 | 0.00 | 0.00 | 0.67 | - | none (best 0.105) |
| 6 | 0 | 0.64 | 3.16 | 0.00 | 0.00 | 0.35 | 0.41 | 0.24 | 0.54 | + | none (best 0.0422) |
| 7 | 0 | 3.2 | 43.8 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00466 | - | constant |

## Groups

- negligible: units 7 (magnitude 0.00466 together; a quarter of the tolerance is 0.019)
- same line: units 0, 5 (gate angle 180 degrees, offset 0.0678)
- same line: units 3, 6 (gate angle 180 degrees, offset -0.451)
- the rest: units 1, 2, 4
