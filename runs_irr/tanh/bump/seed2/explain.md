# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (exp, 0 correction terms, 20 nodes), max error 2.7% of the output range (the check allows 2%):

```python
def net(x0, x1):
    return exp(-x0 * x0 - x1 * x1)
```

Where it is off (formula minus net, % of the output range):
- largest error 2.7%, inside the box (x0 = 2.28, x1 = 0.597)
- by distance from (0, 0), rings 0.423 wide, mean: -0.4 +0.6 +0.1 -0.6 -0.1 +0.4 +0.1 -0.1 -0.0 -0.3; largest: +1.1 +1.3 +1.5 +1.7 +2.3 +2.7 +2.4 +2.5 +2.6 +2.3
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.6 +0.3 -0.3 -0.6 -0.5 -0.5 -0.2 -0.0 -0.1 +0.2 +0.6 +0.6; largest: +2.7 +2.4 +2.3 +2.5 +2.2 +1.5 +1.1 +1.8 +1.9 +1.9 +2.6 +2.5

## Faithful fit

The exp formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no exp formula follows).

Faithful:

```python
def net(x0, x1):
    return 0.9973 * exp(-1.002 * x0 * x0 - 0.9866 * x1 * x1 - 0.004435 * x0 * x1 - 0.01093 * x0 + 0.005284 * x1) - 0.000445
```

Clean:

```python
def net(x0, x1):
    return 0.9973 * exp(-1.002 * x0 * x0 - 0.9866 * x1 * x1 - 0.004435 * x0 * x1 - 0.01093 * x0 + 0.005284 * x1) - 0.000445
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.00546, 0.00269), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 2.6%, inside the box (x0 = 2.28, x1 = 0.597)
  - by distance from (-0.00546, 0.00269), rings 0.424 wide, mean: -0.7 +0.6 +0.2 -0.5 -0.1 +0.4 +0.0 -0.1 -0.1 -0.3; largest: +1.3 +1.2 +1.2 +1.3 +2.2 +2.6 +2.4 +2.5 +2.5 +2.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.5 +0.2 -0.3 -0.4 -0.4 -0.5 -0.2 -0.0 -0.0 +0.2 +0.5 +0.5; largest: +2.6 +2.3 +2.4 +2.5 +2.3 +1.5 +1.3 +1.8 +1.9 +1.9 +2.5 +2.4

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0207, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0 | 0.109 | 5.52 | 0.00 | 0.18 | 0.69 | 0.13 | 0.00 | 0.17 | + | none (best 0.00947) |
| 1 | -146 | 0.487 | 1.72 | 0.26 | 0.21 | 0.26 | 0.16 | 0.12 | 0.897 | - | none (best 0.107) |
| 2 | 1 | -0.489 | 1.42 | 0.18 | 0.14 | 0.18 | 0.15 | 0.35 | 0.602 | + | none (best 0.0717) |
| 3 | -56 | 1.57 | 3.3 | 0.18 | 0.39 | 0.38 | 0.06 | 0.00 | 0.568 | + | none (best 0.0497) |
| 4 | -64 | -0.441 | 1.32 | 0.18 | 0.15 | 0.19 | 0.15 | 0.34 | 0.5 | + | none (best 0.0595) |
| 5 | -151 | 0.984 | 2.71 | 0.18 | 0.31 | 0.37 | 0.14 | 0.01 | 0.73 | + | none (best 0.0878) |
| 6 | 98 | -0.45 | 1.31 | 0.20 | 0.14 | 0.17 | 0.14 | 0.35 | 0.464 | + | none (best 0.0553) |
| 7 | 153 | 1.16 | 2.78 | 0.20 | 0.33 | 0.36 | 0.12 | 0.00 | 0.555 | + | none (best 0.0672) |
| 8 | -48 | 0.473 | 1.58 | 0.27 | 0.20 | 0.25 | 0.15 | 0.13 | 0.64 | - | none (best 0.0762) |
| 9 | 147 | 0.495 | 1.56 | 0.29 | 0.19 | 0.23 | 0.15 | 0.13 | 0.693 | - | none (best 0.0825) |
| 10 | 164 | -0.433 | 1.33 | 0.19 | 0.15 | 0.17 | 0.14 | 0.34 | 0.513 | + | none (best 0.0611) |
| 11 | 66 | -0.449 | 1.34 | 0.18 | 0.15 | 0.18 | 0.15 | 0.34 | 0.514 | + | none (best 0.061) |
| 12 | 83 | 0.449 | 1.3 | 0.36 | 0.14 | 0.16 | 0.14 | 0.20 | 0.455 | - | none (best 0.0541) |
| 13 | -122 | -1.81 | 2.86 | 0.00 | 0.05 | 0.31 | 0.35 | 0.29 | 0.463 | - | none (best 0.0389) |
| 14 | -131 | -0.452 | 1.53 | 0.14 | 0.15 | 0.24 | 0.19 | 0.28 | 0.573 | + | none (best 0.0683) |
| 15 | -161 | -0.418 | 1.37 | 0.19 | 0.15 | 0.18 | 0.15 | 0.33 | 0.534 | + | none (best 0.0636) |

## Groups

- ring: 4 units (1, 8, 9, 12) with gates at distance 0.45-0.49, switch width ~1.6, magnitude ~0.67 each, negative
- the rest: units 0, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15
