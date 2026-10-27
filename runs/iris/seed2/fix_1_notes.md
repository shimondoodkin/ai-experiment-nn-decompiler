# What I changed and why

## Change
Only the second test (medium vs large flowers) changed. The first test (small flowers, class 0) was
already right on every case and is untouched.

    before: large_score = x3 + 0.55 * x2 - 0.4 * x1 - 0.2 * x0;  large if > 1.9
    after:  large_score = x3 + 1.0  * x2 - 0.4 * x1 - 0.2 * x0;  large if > 4.17

So petal length (x2) now counts as much as petal width (x3). The sepal terms stay the same, and the
threshold moves to the middle of the new gap between the classes.

## Why
All three network errors are medium flowers (label 1) called large. They sit just above the old
threshold (scores 1.93 to 2.13), mixed in with real large flowers. Their petals are wide for a medium
flower but not especially long. Giving petal length more weight than the network does pulls two of
them back below the boundary: (6.3,2.5,4.9,1.5) and (5.9,3.2,4.8,1.8) now score 4.14, and the
nearest large flowers score 4.20 or more. I searched linear boundaries with round coefficients, and
this one, which changes only the x2 weight, left the widest gap.

## Score
`python score.py agent.py`: 99/100 right, 2 of the 3 network errors fixed, 0 right answers broken
(the original program gets 97/100).

## What is not fixed, and why
(6.0, 2.7, 5.1, 1.6) is labelled medium. But it lies between two flowers labelled large: (6.0, 2.2,
5.0, 1.5) and (6.3, 2.8, 5.1, 1.5). It has a longer petal and wider petal than both of them. No
linear boundary on the four inputs puts it with the medium flowers without misclassifying
(6.3, 2.8, 5.1, 1.5), which the network gets right. I checked this with a linear program, and adding
product terms such as petal area did not help either. Fixing it would need a rule that runs against
the trend in petal size and covers only this one point. That is the kind of special case the task
rules out, so I left it as the network has it. It is the known overlap between versicolor and
virginica in Iris.

## Side effect
The boundary is rotated (heavier on petal length). Over uniform random inputs in the spec box the new
program disagrees with the original about 7% of the time. Most of that is in corners with no real
flowers (very long petals that are still narrow), which no labelled case covers.
