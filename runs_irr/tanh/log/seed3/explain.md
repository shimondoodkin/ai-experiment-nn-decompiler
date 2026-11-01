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
    return 1.035 * log(x0 + 0.01836) - 0.03043
```

Clean:

```python
def net(x0):
    return log(x0) + 0.05
```

% of the output range:
- distortion (clean minus faithful): largest 1.8%, inside the box (x0 = 0.524)
  - mean error in 10 equal slices of x0: +1.4 +1.7 +1.5 +1.3 +1.1 +1.0 +0.9 +0.8 +0.7 +0.6
  - largest error per slice: +1.8 +1.8 +1.6 +1.4 +1.2 +1.1 +0.9 +0.8 +0.7 +0.6
- ripple (faithful minus net): largest 0.9%, at the box edge (x0 = 0.1)
  - mean error in 10 equal slices of x0: -0.3 -0.3 -0.2 +0.0 +0.1 +0.3 +0.5 +0.6 +0.6 +0.7
  - largest error per slice: +0.9 +0.4 +0.4 +0.1 +0.2 +0.4 +0.5 +0.6 +0.7 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside log | 0 | 0.01836 | 4.42% |
| scale | 1 | 1.035 | 2.11% |
| offset | 0.05 | -0.03043 | 2.11% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0763, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 180 | -0.0515 | 0.482 | 0.91 | 0.06 | 0.03 | 0.00 | 0.00 | 0.74 | - | none (best 0.136) |
| 1 | 180 | -1.4 | 2.08 | 0.31 | 0.27 | 0.32 | 0.11 | 0.00 | 0.7 | - | none (best 0.0422) |
| 2 | 180 | -2.32 | 14.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0243 | - | constant |
| 3 | 180 | -0.643 | 0.986 | 0.69 | 0.12 | 0.15 | 0.03 | 0.00 | 0.748 | - | none (best 0.0492) |
| 4 | 180 | 16.8 | 22.6 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0302 | + | constant |
| 5 | 180 | -0.0409 | 0.647 | 0.88 | 0.09 | 0.04 | 0.00 | 0.00 | 0.591 | - | none (best 0.092) |
| 6 | 0 | 3.05 | 6.08 | 0.00 | 0.14 | 0.86 | 0.00 | 0.00 | 1.11 | + | none (best 0.0437) |
| 7 | 0 | -3.57 | 10.9 | 0.00 | 0.00 | 0.08 | 0.92 | 0.00 | 0.0723 | - | rounded hinge |

## Groups

- lines: units 2, 4 (each a constant or a line on the box; together one linear term)
- same line: units 0, 5 (gate angle 180 degrees, offset -0.0515)
- the rest: units 1, 3, 6, 7
