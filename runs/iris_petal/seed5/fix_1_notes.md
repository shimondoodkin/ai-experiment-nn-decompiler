# What I changed and why

Original class-2 test: `x0 + x1 > 6.65` (a diagonal line). New test: `x0 > 4.95 or x1 > 1.65`
(two axis-aligned thresholds). The class-0 test and the default class 1 are unchanged.

Why: the failing cases are class-2 points just below the diagonal. Some have a large x0 and a small x1
(5.0,1.5 and 5.1,1.5), and some have a small x0 and a large x1 (4.5,1.7 and 4.8,1.8). In the data, every
class-1 point has x0 <= 4.9 and x1 <= 1.6. The only exceptions are the two noted below. So the real
rule is "x0 at least about 5, or x1 at least about 1.7". I put each threshold halfway between the
classes (4.95 and 1.65).

The old class-2 region fits inside the new one: if x0 <= 4.95 and x1 <= 1.65, then x0 + x1 <= 6.6 < 6.65.
So every point the network called class 2 is still class 2. The new region does not overlap the class-0
region, because x1 > 1.65 with x0 >= 1 gives x0 + 1.75*x1 > 3.85.

Result (`python score.py agent.py`): right on 98/100, fixed 5 of 6 network failures, broke 1.
- The broken case is (4.8, 1.8) labelled 1. The same input also appears twice labelled 2, so no
  program can get all three right. Going with the majority (class 2) is the best possible choice.
- Still wrong: (5.1, 1.6) labelled 1. Its neighbours (5.1, 1.5) and (5.0, 1.5) are labelled 2. Getting it
  right would need a special case for a single point, which I chose not to add.
