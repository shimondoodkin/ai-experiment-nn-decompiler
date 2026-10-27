# What I changed and why

## The failures
`program.py` already gets (5.9, 3.2, 4.8, 1.8) right (label 1). The two remaining errors both sit
where versicolor (1) and virginica (2) overlap, at petal length 5.1:

| x0  | x1  | x2  | x3  | score S | program | label |
|-----|-----|-----|-----|---------|---------|-------|
| 6.3 | 2.8 | 5.1 | 1.5 | 3.32    | 1       | 2     |
| 6.0 | 2.7 | 5.1 | 1.6 | 3.69    | 2       | 1     |

S = -0.4*x0 - 0.7*x1 + x2 + 1.8*x3, and the cut is S > 3.45.

The classes 1 and 2 in `repair.csv` cannot be split by any single linear rule, in any subset of the
features (I checked with an LP). So moving or tilting the `large_petal` threshold cannot fix these
points. The sepal terms in S are what give the wrong answers here. The two points have the same
petal length, and the one with the *thinner* petal is the virginica one. With petal length at 5 or
more, every point in the data with petal width <= 1.5 is virginica (6.0/2.2/5.0/1.5 and 6.3/2.8/5.1/1.5),
and the only point with width 1.6 is versicolor. This is the split a decision tree on Iris finds
(petal width at about 1.55 inside the long-petal, narrow-petal branch).

## The change
I added one region, `overlap_zone`: petal length 4.95 to 5.5 and petal width 1.35 to 1.7. Inside it,
petal width alone decides: below 1.55 gives 2, otherwise 1. Everywhere else the network's rule is
unchanged (setosa test first, then `large_petal`).

The zone is bounded on every side so that it changes as little as possible:
- 4.95 sits halfway between versicolor 4.9 and the shortest long-petal virginica, 5.0.
- Petals of 5.5 or longer fall back to the network, which already calls them virginica.
- 1.7 and above is where the network's rule is right on all data (wide petals).
- Below 1.35 nothing in the data needs a change, so the network's rule stays.

## Result
`python score.py agent.py`: 100/100 right, all 3 network failures fixed, 0 network-correct cases
broken. On 200k uniform random points in the input box, the new program differs from `program.py`
on about 0.8% of them, all inside the overlap zone.

Caveat: only three points in the data pin down this zone (two virginica, one versicolor), so its
exact edges are guesses based on the data.
