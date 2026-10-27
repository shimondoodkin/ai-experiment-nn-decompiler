# What I changed

Only the second test, which separates class 1 (versicolor-like) from class 2 (virginica-like), changed.
The class-0 test is the same as before.

- Before: `2.75*x2 + 5*x3 - x0 - 2*x1 > 10.25`
- After:  `5*x2 + 5*x3 - x0 - 2*x1 > 20.85`

## Why

All three network errors are at the class 1 / class 2 border. The network gave petal length (x2) too little
weight compared with petal width (x3). Two class-2 flowers with long petals but medium width,
(6.3, 2.8, 5.1, 1.5) and (6.0, 3.0, 4.8, 1.8), scored below the threshold. Making petal length count as
much as petal width (5 and 5), and moving the threshold to match, puts them on the class-2 side.
The sepal terms stay the same. On the repair set the new score has a gap of 20.7 to 21.0 between
the classes, and the threshold of 20.85 sits in the middle of it.

## Result

`python score.py agent.py`: right on 99 of 100. It fixes 2 of the 3 network errors and makes no new errors.

## Not fixed

(6.0, 2.7, 5.1, 1.6), labelled 1, stays wrong (predicted 2). It is almost identical to
(6.3, 2.8, 5.1, 1.5), labelled 2. Its petals are the same length and slightly wider, yet its label is
the lower class. A linear-programming check shows that classes 1 and 2 in the repair set cannot be
split by any single straight-line rule. Removing just this point makes them separable with a
simple rule. Removing only the other point needs huge, unreadable weights instead.
Fixing this point would mean a special case for one point, so I left it as a label-overlap case.
