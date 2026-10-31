# Explain

Inputs: x0 in [0, 4]; output: regression.

## Formula

A family program passes the check (9 nodes):

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0):
    return 1.025 * sqrt(x0 + 0.02326) - 0.04615
```

Clean:

```python
def net(x0):
    return sqrt(x0 + 0.01)
```

% of the output range:
- distortion (clean minus faithful): largest 1.1%, inside the box (x0 = 0.252)
  - mean error in 10 equal slices of x0: +0.9 +1.0 +0.8 +0.6 +0.4 +0.2 +0.1 -0.1 -0.2 -0.3
  - largest error per slice: +1.1 +1.0 +0.9 +0.7 +0.5 +0.3 +0.2 +0.1 +0.3 +0.4
- ripple (faithful minus net): largest 0.8%, at the box edge (x0 = 0)
  - mean error in 10 equal slices of x0: -0.0 -0.7 -0.3 -0.2 -0.0 +0.1 +0.2 +0.4 +0.5 +0.7
  - largest error per slice: +0.8 +0.8 +0.5 +0.2 +0.1 +0.2 +0.3 +0.4 +0.6 +0.8

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 0.01 | 0.02326 | 2.81% |
| scale | 1 | 1.025 | 2.68% |
| offset | 0 | -0.04615 | 2.47% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0374, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.557 | 11.4 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.206 | + | rounded hinge |
| 1 | 180 | -1.95 | 11.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.203 | - | line |
| 2 | 0 | -3.6 | 16 | 0.00 | 0.00 | 0.60 | 0.40 | 0.00 | 0.126 | + | rounded hinge |
| 3 | 0 | 0.0159 | 2.68 | 0.00 | 0.00 | 0.25 | 0.42 | 0.33 | 0.406 | + | none (best 0.0391) |
| 4 | 180 | 0.153 | 0.676 | 0.87 | 0.11 | 0.02 | 0.00 | 0.00 | 0.312 | - | none (best 0.0391) |
| 5 | 0 | 3.43 | 8.46 | 0.00 | 0.06 | 0.94 | 0.00 | 0.00 | 0.179 | + | rounded hinge |
| 6 | 180 | 0.184 | 8.19 | 0.00 | 0.28 | 0.72 | 0.00 | 0.00 | 0.27 | - | none (best 0.0141) |
| 7 | 0 | 2.91 | 9.83 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.166 | + | rounded hinge |

## Groups

- lines: units 1 (each a constant or a line on the box; together one linear term)
- same line: units 3, 4, 6 (gate angle 0 degrees, offset 0.0159)
- the rest: units 0, 2, 5, 7
