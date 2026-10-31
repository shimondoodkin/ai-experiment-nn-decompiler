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
    return 1.039 * log(x0 + 0.01937) - 0.03063
```

Clean:

```python
def net(x0):
    return log(x0) + 0.05
```

% of the output range:
- distortion (clean minus faithful): largest 1.8%, inside the box (x0 = 0.497)
  - mean error in 10 equal slices of x0: +1.5 +1.7 +1.4 +1.2 +1.0 +0.9 +0.7 +0.6 +0.5 +0.4
  - largest error per slice: +1.8 +1.8 +1.5 +1.3 +1.1 +0.9 +0.8 +0.7 +0.6 +0.5
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 5)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.2 +0.0 +0.2 +0.3 +0.4 +0.6 +0.7 +0.8
  - largest error per slice: +0.8 +0.6 +0.2 +0.1 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.01937 | 4.63% |
| scale | 1 | 1.039 | 2.35% |
| offset | 0.05 | -0.03063 | 2.11% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0765, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.424 | 1.6 | 0.00 | 0.00 | 0.19 | 0.20 | 0.61 | 0.962 | + | none (best 0.0891) |
| 1 | 180 | -1.42 | 4.06 | 0.00 | 0.42 | 0.58 | 0.00 | 0.00 | 0.632 | - | none (best 0.0406) |
| 2 | 180 | 43 | 52 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0139 | + | constant |
| 3 | 0 | 5.64 | 14.6 | 0.00 | 0.01 | 0.99 | 0.00 | 0.00 | 0.713 | + | none (best 0.0259) |
| 4 | 0 | -0.367 | 12.9 | 0.00 | 0.00 | 0.90 | 0.10 | 0.00 | 0.019 | - | constant |
| 5 | 0 | 1.38 | 4.09 | 0.00 | 0.00 | 0.57 | 0.43 | 0.00 | 0.347 | + | none (best 0.0224) |
| 6 | 180 | -0.0146 | 0.543 | 0.90 | 0.07 | 0.02 | 0.00 | 0.00 | 1.1 | - | none (best 0.171) |
| 7 | 180 | -0.0916 | 1.77 | 0.64 | 0.22 | 0.14 | 0.00 | 0.00 | 0.105 | - | rounded hinge |

## Groups

- negligible: units 2, 4 (magnitude 0.0329 together; a quarter of the tolerance is 0.0191)
- same line: units 1, 5 (gate angle 180 degrees, offset -1.42)
- same line: units 6, 7 (gate angle 180 degrees, offset -0.0146)
- the rest: units 0, 3
