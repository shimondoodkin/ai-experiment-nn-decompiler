# What I changed

Only the class-2 boundary. The class-0 rule is the same as before.

Before: class 2 if `x0 + x1 > 6.65`.
After:  class 2 if `x0 + x1 > 6.65 or x0 + 2.5 * x1 > 8.725`.

## Why

All six failures are where classes 1 and 2 overlap (x0 4.5-5.1, x1 1.5-1.8). Five of them are
class-2 points that the net calls 1: (4.5,1.7), (5.0,1.5), (5.1,1.5) and (4.8,1.8) twice. That
pattern says the 1/2 boundary depends more on x1 than the 45-degree diagonal allows. Adding a
steeper line (x1 weighted 2.5x) moves those points to class 2. The correct class-1 points stay on
the class-1 side: (4.7,1.6) scores 8.70 and (4.9,1.5) scores 8.65, both under 8.725. The margin is
about 0.025 on each side, which is the best you can get on a 0.1 grid.

I kept the old diagonal as part of an OR, so every point the network put in class 2 is still
class 2. The two lines cross at about (5.27, 1.38). Below that the old diagonal decides,
and above it the new line pushes the boundary toward smaller x0.

## What is left

- (5.1,1.6), label 1, net 2: still wrong. It sits between (5.0,1.5) and (5.1,1.5), which are
  both class 2, and (4.9,1.8), which is also class 2. No simple boundary fixes it without breaking
  those, so I did not add a special case for it.
- (4.8,1.8): this input appears three times, twice with label 2 and once with label 1. No
  function can get all three right. The new rule follows the majority (2). That fixes two
  failures and breaks the one label-1 copy, which the network had right.

Score: 98/100 right. 5 of 6 network failures fixed, 1 of 94 network-right points broken
(the conflicting duplicate).
