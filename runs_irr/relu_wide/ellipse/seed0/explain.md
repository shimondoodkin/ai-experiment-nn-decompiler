# Explain

Inputs: x0 in [-3, 3], x1 in [-3, 3]; output: binary.

## Formula

A conic program passes the check (28 nodes):

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

## Faithful fit

The conic with every fitted value kept (4 significant digits), next to its clean version.

Faithful:

```python
def net(x0, x1):
    return 0.2494 * x0 * x0 + x1 * x1 - 0.001266 * x0 * x1 - 0.2453 * x0 + 0.008334 * x1 < 0.9325
```

Clean:

```python
def net(x0, x1):
    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95
```

Share of the box where they decide differently: clean and faithful 0.33% (the dirt), faithful and the net 0.04% (what no conic follows), clean and the net 0.33%.

The dirt: each fitted value against its clean one. Effect: the largest change of the left side over the box when only that value is changed (for the threshold, its change), % of the clean threshold.

| term | clean | faithful | effect |
|---|---|---|---|
| x1 | 0 | 0.008334 | 2.63% |
| threshold | 0.95 | 0.9325 | 1.84% |
| x0 | -0.25 | -0.2453 | 1.48% |
| x0·x1 | 0 | -0.001266 | 1.19% |
| x0² | 0.25 | 0.2494 | 0.57% |

## Units

Each relu unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), the sign of its output weight, and the simplest stand-in within a quarter of the tolerance (13.1, 2% of the output range).

| unit | angle | offset | width | off | active | magnitude | sign | stand-in |
|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 0.609 | 0 | 0.61 | 0.39 | 34.1 | - | hinge |
| 1 | -87 | 0.418 | 0 | 0.57 | 0.43 | 44.8 | - | hinge |
| 2 | 117 | 0.378 | 0 | 0.57 | 0.43 | 56.9 | - | hinge |
| 3 | 94 | 0.399 | 0 | 0.57 | 0.43 | 23.2 | - | hinge |
| 4 | 97 | 0.307 | 0 | 0.55 | 0.45 | 55.5 | - | hinge |
| 5 | 174 | 0.453 | 0 | 0.58 | 0.42 | 21.1 | - | hinge |
| 6 | -146 | 0.653 | 0 | 0.63 | 0.37 | 6.86 | - | hinge |
| 7 | -123 | 0.442 | 0 | 0.59 | 0.41 | 58.3 | - | hinge |
| 8 | 86 | 0.402 | 0 | 0.57 | 0.43 | 32.5 | - | hinge |
| 9 | -173 | 0.579 | 0 | 0.60 | 0.40 | 9.12 | - | hinge |
| 10 | 91 | 0.475 | 0 | 0.58 | 0.42 | 17.5 | - | hinge |
| 11 | 116 | 0.508 | 0 | 0.60 | 0.40 | 14.9 | - | hinge |
| 12 | 134 | -11.3 | 0 | 0.00 | 1.00 | 7.19 | + | line |
| 13 | -68 | -17.4 | 0 | 0.00 | 1.00 | 3.03 | + | constant |
| 14 | -163 | -5.5 | 0 | 0.00 | 1.00 | 3.05 | + | constant |
| 15 | 130 | 0.435 | 0 | 0.60 | 0.40 | 24.3 | - | hinge |
| 16 | -154 | 0.44 | 0 | 0.58 | 0.42 | 44 | - | hinge |
| 17 | -79 | 0.482 | 0 | 0.58 | 0.42 | 40 | - | hinge |
| 18 | 114 | 0.439 | 0 | 0.58 | 0.42 | 21.7 | - | hinge |
| 19 | 79 | 0.464 | 0 | 0.58 | 0.42 | 30.6 | - | hinge |
| 20 | -69 | 0.54 | 0 | 0.60 | 0.40 | 52.6 | - | hinge |
| 21 | -71 | 0.51 | 0 | 0.59 | 0.41 | 39.4 | - | hinge |
| 22 | -112 | 0.345 | 0 | 0.56 | 0.44 | 70.3 | - | hinge |
| 23 | 48 | 0.845 | 0 | 0.68 | 0.32 | 43.8 | - | hinge |
| 24 | -120 | 0.38 | 0 | 0.57 | 0.43 | 40.4 | - | hinge |
| 25 | 91 | 0.659 | 0 | 0.61 | 0.39 | 3.54 | - | constant |
| 26 | 86 | -5.82 | 0 | 0.00 | 1.00 | 15.8 | + | line |
| 27 | -46 | 0.823 | 0 | 0.68 | 0.32 | 36.5 | - | hinge |
| 28 | -94 | 0.425 | 0 | 0.57 | 0.43 | 14.7 | - | hinge |
| 29 | -150 | 0.594 | 0 | 0.61 | 0.39 | 17 | - | hinge |
| 30 | 42 | 0.901 | 0 | 0.69 | 0.31 | 30.4 | - | hinge |
| 31 | 122 | 0.403 | 0 | 0.58 | 0.42 | 70.4 | - | hinge |
| 32 | -39 | 0.927 | 0 | 0.69 | 0.31 | 23.4 | - | hinge |
| 33 | 119 | 4.01 | 0 | 1.00 | 0.00 | 0.000214 | - | constant |
| 34 | 59 | 0.704 | 0 | 0.64 | 0.36 | 40.7 | - | hinge |
| 35 | -131 | 0.43 | 0 | 0.59 | 0.41 | 59.6 | - | hinge |
| 36 | 78 | 0.425 | 0 | 0.57 | 0.43 | 31.6 | - | hinge |
| 37 | 75 | 0.506 | 0 | 0.59 | 0.41 | 25.4 | - | hinge |
| 38 | -76 | 0.472 | 0 | 0.58 | 0.42 | 39.9 | - | hinge |
| 39 | -158 | -22.7 | 0 | 0.00 | 1.00 | 2.71 | + | constant |
| 40 | 134 | 0.392 | 0 | 0.59 | 0.41 | 64.7 | - | hinge |
| 41 | -34 | 1.16 | 0 | 0.73 | 0.27 | 9.49 | - | hinge |
| 42 | 83 | -8.53 | 0 | 0.00 | 1.00 | 10.3 | + | line |
| 43 | -63 | 0.678 | 0 | 0.63 | 0.37 | 33.9 | - | hinge |
| 44 | 102 | -32.9 | 0 | 0.00 | 1.00 | 2.48 | + | constant |
| 45 | 107 | 0.52 | 0 | 0.59 | 0.41 | 9.33 | - | hinge |
| 46 | -3 | 1.22 | 0 | 0.70 | 0.30 | 19.4 | - | hinge |
| 47 | -141 | 0.549 | 0 | 0.62 | 0.38 | 10.7 | - | hinge |
| 48 | 59 | 0.771 | 0 | 0.65 | 0.35 | 34.9 | - | hinge |
| 49 | -170 | 0.515 | 0 | 0.59 | 0.41 | 13.6 | - | hinge |
| 50 | 74 | -5.76 | 0 | 0.00 | 1.00 | 12 | + | line |
| 51 | -105 | 0.399 | 0 | 0.57 | 0.43 | 83.6 | - | hinge |
| 52 | 147 | 0.517 | 0 | 0.60 | 0.40 | 19.9 | - | hinge |
| 53 | -96 | 0.485 | 0 | 0.58 | 0.42 | 18.3 | - | hinge |
| 54 | 90 | -5.97 | 0 | 0.00 | 1.00 | 14.7 | + | line |
| 55 | 103 | 0.402 | 0 | 0.57 | 0.43 | 22.3 | - | hinge |
| 56 | 104 | 0.429 | 0 | 0.58 | 0.42 | 29.5 | - | hinge |
| 57 | -80 | 0.565 | 0 | 0.59 | 0.41 | 17 | - | hinge |
| 58 | 107 | 0.31 | 0 | 0.56 | 0.44 | 66.4 | - | hinge |
| 59 | 66 | 0.607 | 0 | 0.61 | 0.39 | 50 | - | hinge |
| 60 | 5 | 4.14 | 0 | 1.00 | 0.00 | 0 | + | constant |
| 61 | 109 | 0.414 | 0 | 0.58 | 0.42 | 15.9 | - | hinge |
| 62 | -151 | 0.763 | 0 | 0.65 | 0.35 | 5.51 | - | constant |
| 63 | -105 | 0.669 | 0 | 0.61 | 0.39 | 5.25 | - | constant |

## Groups

- negligible: units 13, 14, 33, 39, 44, 60 (magnitude 11.3 together; a quarter of the tolerance is 3.27)
- lines: units 12, 25, 26, 42, 50, 54, 62, 63 (each a constant or a line on the box; together one linear term)
- ring: 18 units (1, 5, 10, 11, 15, 16, 17, 18, 19, 21, 28, 36, 37, 38, 49, 52, 53, 56) with gates at distance 0.42-0.52, switch width ~0, magnitude ~25 each, negative
- ring: 6 units (0, 29, 34, 43, 57, 59) with gates at distance 0.57-0.70, switch width ~0, magnitude ~34 each, negative
- same line: units 2, 3, 4, 8, 31, 45, 55, 58, 61 (gate angle 117 degrees, offset 0.378)
- same line: units 6, 7, 22, 24, 35, 47, 51 (gate angle -146 degrees, offset 0.653)
- same line: units 23, 30 (gate angle 48 degrees, offset 0.845)
- same line: units 27, 32, 41 (gate angle -46 degrees, offset 0.823)
- the rest: units 9, 20, 40, 46, 48
