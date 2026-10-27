# What I changed and why

## Change
Only the main class-2 (virginica-like) test was changed. The setosa test and the second
("kink") class-2 test are the same as before.

    before: big_petals = x2 + 1.75*x3 - 0.4*x0 - 0.7*x1 - 10/3
    after:  big_petals = x2 + 1.1*x3  - 0.2*x0 - 0.45*x1 - 4.2

Score: 99/100 correct. 2 of the 3 network failures are fixed, and none of the 97 correct answers
are broken (the original program gets 97/100).

## Why
All three failures sit on the versicolor/virginica border:
- 6.3,2.8,5.1,1.5 (virginica, net said 1): big_petals was -0.088, just below zero.
- 5.9,3.2,4.8,1.8 (versicolor, net said 2): big_petals was +0.017, just above zero.
- 6.0,2.7,5.1,1.6 (versicolor, net said 2).

The network's rule weights petal width too heavily (1.75). That is why the wide-petal but
fairly short 5.9,3.2,4.8,1.8 counted as virginica and the narrower 6.3,2.8,5.1,1.5 did not. The new
rule mostly uses petal length, gives petal width about the same weight as length, and subtracts
a smaller sepal allowance. On the repair set it separates every versicolor/virginica pair except
one. To choose it, I searched coefficients on a 0.05 grid for the widest gap between the two classes,
keeping the kink test because it changes the original decision regions less (about 2.8% of a
uniform sample of the input box changes class, against about 6% if this test replaced both).

## What is not fixed, and why
6.0,2.7,5.1,1.6 (versicolor) is left wrong on purpose. It cannot be separated from virginica together
with the other points by any single linear rule, because its near neighbours 6.3,2.8,5.1,1.5 and
6.0,2.2,5.0,1.5 are virginica with *smaller* petals. It lies just outside the convex hull of the
virginica points, so an extra cut could exclude it. But the only such cut has a margin of about
0.006 and gives petal length a negative weight. That would be a special case for one point, not a general
rule, so I did not add it.

The new boundary is still close to some points: the gap between the classes is about 0.065 in
units of petal length.
