# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 3.9% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 3.9%, at the box edge (x0 = -3, x1 = 2.27)
- by distance from (0, 0), rings 0.423 wide, mean: -0.1 +0.1 -0.4 -1.0 -0.6 -0.0 -0.2 -0.6 -0.8 -0.8; largest: +0.5 +0.8 +1.9 +2.9 +3.0 +2.6 +2.3 +3.8 +3.9 +3.8
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -0.5 -0.9 -0.5 -0.4 -0.7 -0.4 -0.0 -0.2 -0.3 -0.3 -0.4 -0.3; largest: +3.2 +3.8 +3.9 +2.2 +3.9 +2.7 +2.5 +2.3 +2.7 +1.6 +2.6 +2.7

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.9881 * exp(-0.9879 * x0 * x0 - 0.9822 * x1 * x1 + 0.00543 * x0 * x1 + 0.01217 * x0 + 0.01023 * x1) + 0.00533
```

Clean:

```python
def net(x0, x1):
    return 0.9881 * exp(-0.9879 * x0 * x0 - 0.9822 * x1 * x1 + 0.00543 * x0 * x1 + 0.01217 * x0 + 0.01023 * x1) + 0.00533
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (0.00617, 0.00522), rings 0.423 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 3.4%, at the box edge (x0 = -3, x1 = 2.27)
  - by distance from (0.00617, 0.00522), rings 0.423 wide, mean: -0.5 +0.3 +0.3 -0.3 +0.1 +0.5 +0.3 -0.1 -0.2 -0.3; largest: +1.3 +1.7 +1.6 +2.2 +2.6 +2.7 +2.2 +3.3 +3.4 +3.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.1 -0.2 +0.2 +0.2 -0.2 +0.1 +0.4 +0.3 +0.1 +0.2 +0.1 +0.3; largest: +2.7 +3.3 +3.3 +1.9 +3.4 +2.4 +2.7 +2.4 +3.2 +1.8 +2.6 +2.2

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0206, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -32 | 0.507 | 1.31 | 0.34 | 0.16 | 0.20 | 0.14 | 0.17 | 0.561 | - | none (best 0.0666) |
| 1 | 87 | -1.45 | 1.86 | 0.00 | 0.14 | 0.24 | 0.19 | 0.43 | 0.257 | - | none (best 0.0272) |
| 2 | 54 | 0.891 | 3.71 | 0.05 | 0.34 | 0.50 | 0.10 | 0.00 | 0.41 | + | none (best 0.0398) |
| 3 | -149 | 1.18 | 3.38 | 0.11 | 0.37 | 0.43 | 0.09 | 0.00 | 0.231 | + | none (best 0.0218) |
| 4 | -51 | -0.541 | 1.31 | 0.16 | 0.13 | 0.20 | 0.17 | 0.34 | 0.492 | + | none (best 0.0585) |
| 5 | -138 | 1.4 | 3.45 | 0.13 | 0.39 | 0.41 | 0.07 | 0.00 | 0.286 | + | none (best 0.0258) |
| 6 | 87 | 0.496 | 1.44 | 0.35 | 0.15 | 0.18 | 0.15 | 0.18 | 0.702 | - | none (best 0.0836) |
| 7 | 46 | 0.488 | 1.3 | 0.33 | 0.17 | 0.20 | 0.13 | 0.17 | 0.541 | - | none (best 0.0644) |
| 8 | -71 | 0.544 | 1.36 | 0.35 | 0.15 | 0.18 | 0.15 | 0.16 | 0.617 | - | none (best 0.0733) |
| 9 | 127 | 1.69 | 2.83 | 0.27 | 0.36 | 0.31 | 0.06 | 0.00 | 0.449 | + | none (best 0.0447) |
| 10 | 168 | 0.504 | 1.26 | 0.37 | 0.13 | 0.16 | 0.13 | 0.20 | 0.51 | - | none (best 0.0605) |
| 11 | 65 | -0.562 | 1.4 | 0.15 | 0.15 | 0.19 | 0.16 | 0.34 | 0.631 | + | none (best 0.0752) |
| 12 | 28 | -0.508 | 1.33 | 0.17 | 0.15 | 0.19 | 0.16 | 0.34 | 0.527 | + | none (best 0.0626) |
| 13 | 170 | -1.26 | 3.76 | 0.00 | 0.05 | 0.47 | 0.40 | 0.08 | 0.469 | - | none (best 0.0276) |
| 14 | 145 | -0.109 | 4 | 0.00 | 0.19 | 0.58 | 0.23 | 0.00 | 0.093 | - | none (best 0.0114) |
| 15 | -172 | -0.525 | 1.35 | 0.18 | 0.14 | 0.17 | 0.14 | 0.36 | 0.562 | + | none (best 0.0668) |

## Groups

- ring: 5 units (0, 6, 7, 8, 10) with gates at distance 0.49-0.54, switch width ~1.3, magnitude ~0.56 each, negative
- the rest: units 1, 2, 3, 4, 5, 9, 11, 12, 13, 14, 15
