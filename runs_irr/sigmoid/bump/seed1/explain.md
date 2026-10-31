# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 4.7% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 4.7%, at the box edge (x0 = -0.362, x1 = -3)
- by distance from (0, 0), rings 0.423 wide, mean: +0.1 +0.2 -0.5 -1.1 -0.5 +0.1 -0.2 -0.5 -0.6 -0.7; largest: +1.1 +1.4 +2.1 +3.0 +3.0 +2.7 +4.0 +4.7 +3.0 +3.4
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.4 -0.1 -0.4 -0.1 -0.2 -0.2 -0.4 -0.5 -0.4 -0.5 -0.8 -0.5; largest: +3.5 +3.0 +3.7 +4.1 +3.4 +2.1 +2.4 +3.2 +4.7 +2.8 +3.0 +3.0

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 1.014 * exp(-1.094 * x0 * x0 - 1.022 * x1 * x1 + 0.03199 * x0 * x1 + 0.0531 * x0 - 0.01696 * x1) + 0.005581
```

Clean:

```python
def net(x0, x1):
    return 1.014 * exp(-1.094 * x0 * x0 - 1.022 * x1 * x1 + 0.03199 * x0 * x1 + 0.0531 * x0 - 0.01696 * x1) + 0.005581
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.0242, -0.00792), rings 0.426 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 4.2%, at the box edge (x0 = 0.735, x1 = 2.99)
  - by distance from (0.0242, -0.00792), rings 0.426 wide, mean: +1.6 +0.1 -1.5 -1.7 -0.4 +0.5 +0.3 +0.0 -0.0 -0.2; largest: +3.3 +3.7 +3.6 +3.6 +2.8 +3.1 +3.8 +4.2 +3.5 +3.5
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.1 +0.3 +0.0 +0.1 -0.3 -0.7 -0.8 -0.3 +0.1 +0.1 -0.4 -0.3; largest: +3.2 +3.5 +4.2 +3.6 +3.6 +3.7 +3.5 +2.7 +4.2 +3.3 +3.3 +3.3

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0207, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 21 | -1.19 | 2.69 | 0.00 | 0.12 | 0.35 | 0.30 | 0.23 | 0.307 | - | none (best 0.0367) |
| 1 | 116 | 0.468 | 1.19 | 0.37 | 0.14 | 0.17 | 0.13 | 0.19 | 0.443 | - | none (best 0.0527) |
| 2 | 155 | 0.572 | 1.33 | 0.36 | 0.16 | 0.18 | 0.15 | 0.16 | 0.627 | - | none (best 0.0747) |
| 3 | 81 | 0.492 | 1.27 | 0.37 | 0.13 | 0.16 | 0.13 | 0.20 | 0.445 | - | none (best 0.053) |
| 4 | 112 | -0.932 | 12.1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.122 | + | rounded hinge |
| 5 | -82 | 0.493 | 1.29 | 0.37 | 0.13 | 0.16 | 0.14 | 0.20 | 0.484 | - | none (best 0.0576) |
| 6 | -66 | 1.79 | 5.05 | 0.02 | 0.46 | 0.52 | 0.00 | 0.00 | 0.362 | + | none (best 0.0233) |
| 7 | 169 | 0.786 | 2.83 | 0.15 | 0.30 | 0.36 | 0.19 | 0.00 | 0.597 | + | none (best 0.0651) |
| 8 | -122 | -2.11 | 2.3 | 0.00 | 0.04 | 0.22 | 0.28 | 0.46 | 0.366 | - | none (best 0.0358) |
| 9 | 0 | 0.522 | 1.5 | 0.33 | 0.16 | 0.19 | 0.16 | 0.16 | 0.597 | - | none (best 0.0712) |
| 10 | -137 | -0.505 | 1.24 | 0.17 | 0.12 | 0.19 | 0.17 | 0.34 | 0.431 | + | none (best 0.0512) |
| 11 | -42 | 1.5 | 2.91 | 0.22 | 0.37 | 0.34 | 0.08 | 0.00 | 0.432 | + | none (best 0.0457) |
| 12 | 177 | 0.587 | 1.26 | 0.39 | 0.13 | 0.16 | 0.13 | 0.19 | 0.276 | - | none (best 0.0328) |
| 13 | 24 | -0.584 | 1.32 | 0.16 | 0.14 | 0.18 | 0.15 | 0.36 | 0.581 | + | none (best 0.0692) |
| 14 | 135 | -0.564 | 1.46 | 0.14 | 0.13 | 0.23 | 0.19 | 0.31 | 0.609 | + | none (best 0.0725) |
| 15 | 61 | -0.482 | 1.28 | 0.18 | 0.14 | 0.18 | 0.15 | 0.35 | 0.5 | + | none (best 0.0594) |

## Groups

- ring: 6 units (1, 2, 3, 5, 9, 12) with gates at distance 0.47-0.59, switch width ~1.3, magnitude ~0.46 each, negative
- the rest: units 0, 4, 6, 7, 8, 10, 11, 13, 14, 15
