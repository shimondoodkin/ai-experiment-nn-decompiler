# Explain

Inputs: x0 in [-2, 2], x1 in [-2, 2]; output: regression.

## Formula

A family program passes the check (24 nodes):

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1.25) - 0.07
```

## Faithful fit

The sqrt formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no sqrt formula follows).

Faithful:

```python
def net(x0, x1):
    return sqrt(0.9025 * x0 * x0 + 0.9136 * x1 * x1 - 0.003122 * x0 * x1 + 0.0001016 * x0 + 0.0003983 * x1 + 0.7402) + 0.1387
```

Clean:

```python
def net(x0, x1):
    return sqrt(x0 * x0 + x1 * x1 + 1.25) - 0.07
```

% of the output range:
- distortion (clean minus faithful): largest 2.5%, inside the box (x0 = -0.00136, x1 = 0.00599)
  - by distance from (0, 0), rings 0.282 wide, mean: +2.3 +1.8 +1.1 +0.4 -0.1 -0.3 -0.4 -0.4 -0.2 -0.0; largest: +2.5 +2.2 +1.5 +0.8 +0.4 +0.7 +0.7 +0.7 +0.6 +0.3
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.2 +0.0 -0.1 -0.1 -0.1 +0.2 +0.2 +0.0 -0.0 -0.1 -0.1 +0.2; largest: +2.5 +2.5 +2.5 +2.5 +2.5 +2.5 +2.5 +2.5 +2.5 +2.5 +2.5 +2.5
- ripple (faithful minus net): largest 1.2%, inside the box (x0 = 1.6, x1 = 1.68)
  - by distance from (-5.67e-05, -0.000218), rings 0.282 wide, mean: -1.0 -0.3 +0.5 +0.9 +0.7 +0.2 -0.3 -0.4 -0.7 -0.4; largest: +1.2 +0.9 +1.1 +1.2 +1.2 +0.8 +1.1 +1.2 +1.2 +1.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 -0.1 +0.3 +0.4 -0.1 +0.3 -0.1 -0.1 +0.3 +0.4 -0.1 -0.1; largest: +1.2 +1.2 +1.2 +1.2 +1.2 +1.2 +1.2 +1.2 +1.2 +1.2 +1.2 +1.2

The dirt: each fitted value against its clean one. Effect: the largest change over the box when only that value is changed, % of the output range (changes can offset each other).

| term | clean | faithful | effect |
|---|---|---|---|
| constant inside sqrt | 1.25 | 0.7402 | 13.12% |
| offset | -0.07 | 0.1387 | 10.63% |
| x0² inside sqrt | 1 | 0.9025 | 4.42% |
| x1² inside sqrt | 1 | 0.9136 | 3.91% |
| x0·x1 inside sqrt | 0 | -0.003122 | 0.10% |
| x1 inside sqrt | 0 | 0.0003983 | 0.01% |
| x0 inside sqrt | 0 | 0.0001016 | 0.00% |

## Units

Each sigmoid unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.0392, 2% of the output range).

| unit | angle | offset | width | off | low knee | middle | high knee | on | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 97 | 1.94 | 2.98 | 0.24 | 0.47 | 0.29 | 0.00 | 0.00 | 0.365 | + | none (best 0.0326) |
| 1 | -98 | 1.78 | 1.99 | 0.45 | 0.32 | 0.24 | 0.00 | 0.00 | 0.681 | + | none (best 0.0529) |
| 2 | -178 | 1.52 | 2.57 | 0.23 | 0.41 | 0.36 | 0.00 | 0.00 | 1.06 | + | none (best 0.0825) |
| 3 | 79 | 1.36 | 2.64 | 0.17 | 0.42 | 0.40 | 0.00 | 0.00 | 0.844 | + | none (best 0.0581) |
| 4 | 5 | 1.65 | 2.03 | 0.40 | 0.32 | 0.28 | 0.00 | 0.00 | 0.99 | + | none (best 0.0782) |
| 5 | 173 | -3.41 | 22.9 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0241 | + | line |
| 6 | -173 | -1.03 | 15.2 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0738 | + | line |
| 7 | 138 | -0.461 | 8.76 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0262 | - | line |
| 8 | 33 | 0.554 | 1.97 | 0.12 | 0.32 | 0.41 | 0.14 | 0.00 | 0.222 | + | none (best 0.0268) |
| 9 | -3 | 3.91 | 28.5 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00714 | - | constant |
| 10 | 107 | -0.0637 | 5.41 | 0.00 | 0.02 | 0.95 | 0.03 | 0.00 | 0.038 | - | line |
| 11 | 128 | 1.11 | 2.14 | 0.20 | 0.40 | 0.35 | 0.05 | 0.00 | 0.635 | + | none (best 0.0541) |
| 12 | -112 | 0.727 | 1.95 | 0.17 | 0.33 | 0.38 | 0.12 | 0.00 | 0.632 | + | none (best 0.0689) |
| 13 | -53 | 1.12 | 2.19 | 0.19 | 0.40 | 0.36 | 0.05 | 0.00 | 0.709 | + | none (best 0.0576) |
| 14 | -175 | -0.513 | 1.37 | 0.03 | 0.21 | 0.26 | 0.22 | 0.28 | 0.229 | - | none (best 0.0275) |
| 15 | -177 | -1.74 | 17.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0393 | + | line |

## Groups

- negligible: units 9 (magnitude 0.00714 together; a quarter of the tolerance is 0.00979)
- lines: units 5, 6, 7, 10, 15 (each a constant or a line on the box; together one linear term)
- ring: 4 units (1, 2, 3, 4) with gates at distance 1.36-1.78, switch width ~2.3, magnitude ~0.92 each, positive
- the rest: units 0, 8, 11, 12, 13, 14
