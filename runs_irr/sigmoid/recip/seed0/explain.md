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
    return 1.031 / (x0 + 0.01419) - 0.01333
```

Clean:

```python
def net(x0):
    return 1 / x0
```

% of the output range:
- distortion (clean minus faithful): largest 0.5%, at the box edge (x0 = 0.5)
  - mean error in 10 equal slices of x0: +0.0 -0.2 -0.1 +0.0 +0.1 +0.2 +0.2 +0.3 +0.3 +0.4
  - largest error per slice: +0.5 +0.2 +0.1 +0.1 +0.1 +0.2 +0.3 +0.3 +0.3 +0.4
- ripple (faithful minus net): largest 0.4%, at the box edge (x0 = 0.589)
  - mean error in 10 equal slices of x0: -0.1 +0.3 +0.0 -0.0 -0.1 -0.1 -0.2 -0.3 -0.3 -0.3
  - largest error per slice: +0.4 +0.3 +0.2 +0.1 +0.1 +0.2 +0.2 +0.3 +0.3 +0.3

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| scale | 1 | 1.031 | 3.57% |
| constant inside reciprocal | 0 | 0.01419 | 3.18% |
| offset | 0 | -0.01333 | 0.77% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0347, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.332 | 2.99 | 0.19 | 0.54 | 0.27 | 0.00 | 0.00 | 0.472 | + | none (best 0.0454) |
| 1 | 180 | -4.02 | 21 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0574 | + | line |
| 2 | 180 | -0.159 | 0.947 | 0.82 | 0.17 | 0.00 | 0.00 | 0.00 | 0.171 | + | none (best 0.0232) |
| 3 | 180 | -0.292 | 0.632 | 0.88 | 0.12 | 0.01 | 0.00 | 0.00 | 0.274 | + | none (best 0.0348) |
| 4 | 180 | -0.157 | 1.12 | 0.78 | 0.20 | 0.02 | 0.00 | 0.00 | 0.33 | + | none (best 0.0411) |
| 5 | 180 | 0.11 | 11.4 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.178 | + | rounded hinge |
| 6 | 0 | 0.431 | 2.11 | 0.00 | 0.00 | 0.21 | 0.38 | 0.42 | 0.253 | - | none (best 0.0271) |
| 7 | 0 | -0.415 | 3.88 | 0.00 | 0.00 | 0.16 | 0.69 | 0.15 | 0.00168 | - | constant |

## Groups

- negligible: units 7 (magnitude 0.00168 together; a quarter of the tolerance is 0.00868)
- lines: units 1 (each a constant or a line on the box; together one linear term)
- same line: units 0, 2, 3, 4, 6 (gate angle 180 degrees, offset -0.332)
- the rest: units 5
