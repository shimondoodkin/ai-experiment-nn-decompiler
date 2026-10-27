# What I changed

Only the second boundary (class 1 vs class 2). The class-0 boundary is unchanged.

- Network: `x0 + 1.1003194*x1 > 6.8039224 -> 2`
- Now:     `x0 + 2.5*x1 > 8.725 -> 2`   (equivalently `0.4*x0 + x1 > 3.49`)

## Why

All 6 failing cases sit on the class 1 / class 2 border. Five of them are class-2 points the network
calls 1: (4.5,1.7), (4.8,1.8) twice, (5.0,1.5), (5.1,1.5). They have relatively wide petals
(x1) for their length, so the network's line leans too much on petal length and too little on petal
width. Tilting the line to weigh x1 more fixes the general pattern. The slope 2.5 and offset 8.725 put the
line midway between the closest class-1 points, (4.9,1.5) and (4.7,1.6), and the closest class-2
points, (5.0,1.5) and (4.5,1.7). Of the slopes that separate those four points, 2.5 leaves the widest gap.
No other point is closer than that.

## Result (score.py)

98/100 right (network: 94). 5 of 6 failures fixed. 1 previously right point is now wrong.

## What remains, and why I left it

- (4.8,1.8) appears three times in repair.csv: twice labelled 2 and once labelled 1. No function can
  get all three right. The new rule gives 2 (the majority), which fixes two cases and breaks the one
  labelled 1.
- (5.1,1.6) label 1 (net says 2) stays wrong. It lies between (5.0,1.5)/(5.1,1.5), both class 2, and
  wider/longer class-2 points. Getting it right would need a carve-out for that one point, which
  the instructions rule out.
