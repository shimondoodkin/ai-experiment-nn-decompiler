# What I changed

The class 0 rule is unchanged. All six failures are in the class 1 / class 2 overlap.

The network's class 2 region was `length + 1.1*width > 6.8 or length + 0.7*width > 6.6`. I replaced it with one line:
`length + 1.1*width > 6.6`. Since width > 0, this line contains the old region, so the second piece of the old
rule is no longer needed. The program only grows class 2. It never shrinks it.

## Why

Four of the failures are class 2 points that the network calls 1: (5.0,1.5), (5.1,1.5), and two copies of (4.8,1.8).
Their scores on `length + 1.1*width` are 6.65, 6.75 and 6.78, all below the old cutoff of 6.8. The highest class 1
point the network gets right scores 6.55, at (4.9,1.5). A cutoff of 6.6 falls in that gap, about halfway. It moves
the boundary between versicolor and virginica a little toward smaller petals. It is not a rule for a single point.

## What is still wrong (3 of 100), and why I left it

- (4.8,1.8) shows up three times: twice labelled 2 and once labelled 1. No function can get all three right.
  The new rule follows the majority (2), so the one copy labelled 1 is the single "broken" point.
- (5.1,1.6) is labelled 1, but (5.0,1.5) and (5.1,1.5) are labelled 2, and they are smaller on both measurements.
  Fixing it would need a non-monotone carve-out around one point.
- (4.5,1.7) is labelled 2, but (4.7,1.6), (4.9,1.5) and others are labelled 1. A line that separates these
  would need a margin of about 0.05, or a special rule like "width >= 1.7". Both would fit a single point (a known
  outlier in iris).

Score: 97/100 right, 4 of 6 failures fixed, 1 of the network's right answers changed (the conflicting duplicate).
