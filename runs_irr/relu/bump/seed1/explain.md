# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (reciprocal, 2 correction terms, 42 nodes), max error 9.8% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = x0 * x0 + x1 * x1 + 0.75
    return 1 / u + 0.01431 * relu(u - 0.75) - 0.25
```

Where it is off (formula minus net, % of the output range):
- largest error 9.7%, inside the box (x0 = 0.947, x1 = -0.475)
- by distance from (0, 0), rings 0.423 wide, mean: +4.3 -2.0 -2.5 +0.7 -0.4 -1.2 -1.3 -0.9 +1.0 +3.7; largest: +8.2 +8.3 +9.7 +9.1 +9.0 +7.6 +6.6 +9.5 +6.9 +8.3
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -1.0 -0.1 -0.3 -0.7 -0.3 -1.7 -1.3 -0.2 -0.8 -1.1 +0.2 -1.4; largest: +8.7 +8.1 +7.7 +6.6 +6.2 +8.7 +8.6 +6.8 +6.3 +7.3 +8.3 +9.7

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return 1 / (1.506 * x0 * x0 + 1.591 * x1 * x1 + 0.03298 * x0 * x1 + 0.03608 * x0 + 0.01231 * x1 + 0.8514) - 0.08595
```

Clean:

```python
def net(x0, x1):
    return 1 / (1.506 * x0 * x0 + 1.591 * x1 * x1 + 0.03298 * x0 * x1 + 0.03608 * x0 + 0.01231 * x1 + 0.8514) - 0.08595
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.0119, -0.00374), rings 0.424 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 11.7%, inside the box (x0 = -1.45, x1 = -0.047)
  - by distance from (-0.0119, -0.00374), rings 0.424 wide, mean: +2.9 -4.4 -2.4 +3.1 +2.9 +1.6 -0.0 -1.9 -2.9 -3.3; largest: +7.0 +9.9 +10.4 +11.7 +9.5 +8.9 +8.7 +11.3 +9.8 +6.2
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.4 -0.4 +0.7 +0.5 -0.3 +0.3 +0.5 -0.3 +0.5 +0.2 +0.1 +0.2; largest: +11.6 +9.7 +9.4 +8.9 +9.5 +11.1 +11.7 +8.5 +6.6 +8.6 +10.6 +11.3

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.023, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -48 | -0.0259 | 0 | 0.49 | 0.51 | 0.511 | - | hinge |
| 1 | 34 | 1.3 | 0 | 0.75 | 0.25 | 1.12 | + | hinge |
| 2 | 164 | 0.171 | 0 | 0.53 | 0.47 | 0.914 | - | hinge |
| 3 | 114 | -0.00512 | 0 | 0.50 | 0.50 | 1.68 | - | hinge |
| 4 | 125 | -4.08 | 0 | 0.00 | 1.00 | 0.744 | - | hinge |
| 5 | 65 | -0.0241 | 0 | 0.50 | 0.50 | 2.66 | - | hinge |
| 6 | 5 | -1.21 | 0 | 0.30 | 0.70 | 0.995 | + | hinge |
| 7 | -79 | 36.8 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 8 | -89 | -1.21 | 0 | 0.30 | 0.70 | 1.7 | + | hinge |
| 9 | 4 | 0.0263 | 0 | 0.51 | 0.49 | 1.22 | - | hinge |
| 10 | 152 | -1.32 | 0 | 0.25 | 0.75 | 2.23 | + | hinge |
| 11 | -55 | -0.0793 | 0 | 0.48 | 0.52 | 0.827 | - | hinge |
| 12 | -154 | 0.189 | 0 | 0.53 | 0.47 | 0.977 | - | hinge |
| 13 | 40 | -1.13 | 0 | 0.27 | 0.73 | 1.86 | + | hinge |
| 14 | 146 | 1.17 | 0 | 0.73 | 0.27 | 1.14 | + | hinge |
| 15 | 92 | -1.2 | 0 | 0.30 | 0.70 | 1.71 | + | hinge |

## Groups

- negligible: units 7 (magnitude 0 together; a quarter of the tolerance is 0.00576)
- same line: units 0, 11 (gate angle -48 degrees, offset -0.0259)
- the rest: units 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15
