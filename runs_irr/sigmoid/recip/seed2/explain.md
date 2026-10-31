# Explain

Inputs: x0 in [0.5, 4]; output: regression.

## Formula

A family program passes the check (6 nodes):

```python
def net(x0):
    return 1 / x0
```

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0):
    return 1.063 / (x0 + 0.03075) - 0.02308
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 1.2%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.1 -0.5 -0.3 -0.2 +0.0 +0.1 +0.3 +0.4 +0.4 +0.5
  - largest error per slice: +1.2 +0.5 +0.4 +0.2 +0.1 +0.2 +0.3 +0.4 +0.5 +0.5
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: -0.2 +0.7 +0.3 -0.0 -0.1 -0.2 -0.2 -0.3 -0.5 -0.6
  - largest error per slice: +0.8 +0.8 +0.6 +0.1 +0.2 +0.2 +0.2 +0.4 +0.5 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.063 | 7.36% |
| constant inside reciprocal | 0 | 0.03075 | 6.77% |
| offset | 0 | -0.02308 | 1.35% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0342, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -6.57 | 36.8 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0346 | + | line |
| 1 | 180 | 0.38 | 3.53 | 0.24 | 0.63 | 0.13 | 0.00 | 0.00 | 0.151 | + | none (best 0.0151) |
| 2 | 0 | 0.0931 | 25 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0102 | + | constant |
| 3 | 180 | -0.372 | 5.71 | 0.00 | 0.42 | 0.58 | 0.00 | 0.00 | 0.291 | + | none (best 0.0187) |
| 4 | 0 | 0.395 | 3.61 | 0.00 | 0.00 | 0.36 | 0.64 | 0.00 | 0.156 | - | none (best 0.0132) |
| 5 | 180 | -0.769 | 2.1 | 0.32 | 0.38 | 0.30 | 0.00 | 0.00 | 0.259 | + | none (best 0.0231) |
| 6 | 180 | -0.255 | 0.929 | 0.80 | 0.17 | 0.03 | 0.00 | 0.00 | 0.28 | + | none (best 0.0347) |
| 7 | 180 | -0.27 | 0.914 | 0.80 | 0.16 | 0.03 | 0.00 | 0.00 | 0.551 | + | none (best 0.0676) |

## Groups

- lines: units 0, 2 (each a constant or a line on the box; together one linear term)
- same line: units 3, 4, 6, 7 (gate angle 180 degrees, offset -0.372)
- the rest: units 1, 5
