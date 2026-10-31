# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: regression.

## Formula

No formula passes the check. Near formula (reciprocal, 3 correction terms, 61 nodes), max error 10.1% of the output range (the check allows 2%):

```python
def net(x0, x1):
    u = 0.75 * x0 * x0 + 0.75 * x1 * x1 + 1
    return 1 / u + 0.2 * relu(u - 2.25) - 0.1771 * relu(u + 5) + 1
```

Where it is off (formula minus net, % of the output range):
- largest error 10.0%, inside the box (x0 = 0.717, x1 = -1.69)
- by distance from (0, 0), rings 0.423 wide, mean: +3.0 +2.6 -2.0 -3.7 -3.3 -2.8 -1.2 -1.0 -2.8 -1.4; largest: +7.8 +7.7 +9.2 +9.9 +10.0 +9.1 +7.5 +9.7 +8.2 +4.1
- by angle, 12 sectors of 30 degrees from the x0 axis, mean: -2.2 -0.6 -1.9 -2.5 -1.1 -2.5 -2.4 -1.3 -2.7 -3.1 -1.9 -2.5; largest: +8.8 +8.5 +9.7 +8.1 +6.7 +7.3 +6.6 +8.2 +8.5 +10.0 +9.1 +9.9

## Faithful fit

The reciprocal formula with every fitted value kept (4 significant digits) and no correction terms, next to its clean version. The net minus the clean formula is the sum of two parts: the distortion (clean minus faithful: the dirt) and the ripple (faithful minus net: what no reciprocal formula follows).

Faithful:

```python
def net(x0, x1):
    return 1 / (1.754 * x0 * x0 + 1.843 * x1 * x1 + 0.09005 * x0 * x1 + 0.07216 * x0 + 0.008332 * x1 + 0.888) - 0.0751
```

Clean:

```python
def net(x0, x1):
    return 1 / (1.754 * x0 * x0 + 1.843 * x1 * x1 + 0.09005 * x0 * x1 + 0.07216 * x0 + 0.008332 * x1 + 0.888) - 0.0751
```

% of the output range:
- distortion (clean minus faithful): largest 0.0%, inside the box (x0 = 0.822, x1 = 0.642)
  - by distance from (-0.0205, -0.00176), rings 0.425 wide, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0; largest: +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.0
- ripple (faithful minus net): largest 11.9%, inside the box (x0 = 0.351, x1 = 0.875)
  - by distance from (-0.0205, -0.00176), rings 0.425 wide, mean: +4.8 -6.4 -6.8 -1.0 +0.4 +0.9 +1.2 -1.0 -6.0 -8.4; largest: +11.4 +11.7 +11.9 +9.4 +11.8 +10.1 +7.1 +9.9 +10.6 +11.7
  - by angle, 12 sectors of 30 degrees from the x0 axis, mean: -1.1 -0.9 -1.0 -1.1 -0.7 -0.4 -0.5 -1.3 -1.6 -1.9 -1.8 -1.1; largest: +11.6 +11.6 +11.9 +9.6 +10.9 +10.5 +11.1 +11.3 +9.8 +11.6 +11.8 +11.8

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (0.022, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | -1 | -0.1 | 0 | 0.48 | 0.52 | 0.686 | - | hinge |
| 1 | -27 | -1.2 | 0 | 0.28 | 0.72 | 1.02 | + | hinge |
| 2 | -29 | -0.121 | 0 | 0.47 | 0.53 | 1.49 | - | hinge |
| 3 | -89 | 1.25 | 0 | 0.71 | 0.29 | 0.835 | + | hinge |
| 4 | -180 | -1.31 | 0 | 0.28 | 0.72 | 1.78 | + | hinge |
| 5 | -61 | 9.82 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 6 | 113 | -0.0243 | 0 | 0.50 | 0.50 | 1.2 | - | hinge |
| 7 | 62 | -0.00357 | 0 | 0.50 | 0.50 | 1.61 | - | hinge |
| 8 | -58 | -0.0775 | 0 | 0.48 | 0.52 | 0.859 | - | hinge |
| 9 | 89 | 1.27 | 0 | 0.71 | 0.29 | 0.816 | + | hinge |
| 10 | -155 | 0.0595 | 0 | 0.51 | 0.49 | 0.954 | - | hinge |
| 11 | 25 | -1.21 | 0 | 0.28 | 0.72 | 1.28 | + | hinge |
| 12 | -3 | -1.25 | 0 | 0.29 | 0.71 | 1.14 | + | hinge |
| 13 | -106 | 0.004 | 0 | 0.50 | 0.50 | 0.308 | - | hinge |
| 14 | -141 | 0.271 | 0 | 0.56 | 0.44 | 0.601 | - | hinge |
| 15 | -162 | 0.118 | 0 | 0.52 | 0.48 | 0.0631 | - | hinge |

## Groups

- negligible: units 5 (magnitude 0 together; a quarter of the tolerance is 0.00551)
- same line: units 6, 8 (gate angle 113 degrees, offset -0.0243)
- same line: units 10, 15 (gate angle -155 degrees, offset 0.0595)
- the rest: units 0, 1, 2, 3, 4, 7, 9, 11, 12, 13, 14
