# Explain

Inputs: x0 in [-1.5, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return log(x0 + 2)
```

## Faithful fit

The log formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no log formula follows).

Faithful:

```python
def net(x0):
    return 1.029 * log(x0 + 2.044) - 0.04998
```

Clean:

```python
def net(x0):
    return log(x0 + 2)
```

% of the output range:
- distortion (clean minus faithful): largest 0.7%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: -0.1 +0.3 +0.3 +0.2 +0.1 +0.0 -0.1 -0.2 -0.3 -0.3
  - largest error per slice: +0.7 +0.3 +0.3 +0.3 +0.2 +0.1 +0.1 +0.2 +0.3 +0.4
- ripple (faithful minus net): largest 0.6%, at the box edge (x0 = -1.5)
  - mean error in 10 equal slices of x0: +0.1 -0.5 -0.2 -0.3 -0.2 +0.0 +0.1 +0.2 +0.2 +0.4
  - largest error per slice: +0.6 +0.6 +0.3 +0.3 +0.3 +0.1 +0.2 +0.2 +0.3 +0.5

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 2 | 2.044 | 3.44% |
| scale | 1 | 1.029 | 2.12% |
| offset | 0 | -0.04998 | 2.04% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.049, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | -0.235 | 2.08 | 0.00 | 0.09 | 0.28 | 0.24 | 0.39 | 0.444 | + | none (best 0.0289) |
| 1 | 0 | 1.48 | 8.17 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.105 | + | line |
| 2 | 0 | 0.198 | 6.71 | 0.00 | 0.00 | 0.77 | 0.23 | 0.00 | 0.338 | + | none (best 0.0173) |
| 3 | 0 | 0.82 | 6.44 | 0.00 | 0.00 | 0.86 | 0.14 | 0.00 | 0.337 | + | none (best 0.0129) |
| 4 | 180 | 1.54 | 1.35 | 0.76 | 0.15 | 0.08 | 0.00 | 0.00 | 0.812 | - | none (best 0.104) |
| 5 | 0 | 2.01 | 7.87 | 0.00 | 0.10 | 0.90 | 0.00 | 0.00 | 0.125 | + | rounded hinge |
| 6 | 0 | 2.56 | 7.91 | 0.00 | 0.20 | 0.80 | 0.00 | 0.00 | 0.135 | + | rounded hinge |
| 7 | 0 | 2.45 | 7.86 | 0.00 | 0.18 | 0.82 | 0.00 | 0.00 | 0.153 | + | rounded hinge |

## Groups

- lines: units 1 (each a constant or a line on the box; together one linear term)
- same line: units 6, 7 (gate angle 0 degrees, offset 2.56)
- the rest: units 0, 2, 3, 4, 5
