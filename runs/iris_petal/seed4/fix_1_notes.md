# What changed and why

The class 0 rule (`x0 + x1 < 3.3`) and the original class 1/2 line (`x0 + 1.1*x1 >= 6.75 -> 2`) are unchanged.
I added one more condition: a point is also class 2 when `x0 + 2.5*x1 >= 8.72`.

**Why:** the failing cases sit in the iris versicolor/virginica overlap, at petal length 4.5-5.1 and width 1.5-1.8.
- The network says class 1 for points labelled 2: (4.5, 1.7), (5.0, 1.5), (5.1, 1.5), and (4.8, 1.8) twice.
- The original line gives too little weight to petal width. It puts (4.5, 1.7), which is class 2, on the same side as
  (4.7, 1.6) and (4.9, 1.5), which are class 1.
- The class 2 points in this area satisfy `x0 + 2.5*x1 >= 8.75`. The nearest class 1 points reach 8.70 at (4.7, 1.6) and
  8.65 at (4.9, 1.5). A threshold of 8.72 splits that gap.
- To keep the change small, the new line is ORed with the old one rather than replacing it. The two lines cross at about
  (5.2, 1.41). So the change only adds a thin wedge of class 2 for width > 1.41, and everything else behaves as before.

**Score:** 98 of 100 right; 5 of the network's 6 failures fixed; 1 of its right answers broken.
- **Still wrong:** (5.1, 1.6) is labelled 1, but (5.0, 1.5) and (5.1, 1.5) are labelled 2. Getting all three right would
  need class 1 to come back as width increases, which a monotone rule cannot do, so I left it as class 2 (noise).
- **Broken:** (4.8, 1.8) appears three times, labelled 2, 2 and 1. The rule follows the majority (2), so the copy
  labelled 1 is now wrong. The given program.py already says 2 at this point, because it falls just past its line at 6.78.
