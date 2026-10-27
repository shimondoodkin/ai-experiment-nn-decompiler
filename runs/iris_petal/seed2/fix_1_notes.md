# Changes to program.py

Only the "large" (class 2) condition changed; the class-0 rule is untouched.

Before: `large = x0 + 1.1*x1 > 6.75 or x0 > 6.3`
After:  `large = x0 + 1.1*x1 > 6.6 or x1 > 1.65 or x0 > 6.3`

Why:
- Failing (5.0, 1.5) and (5.1, 1.5) should be class 2 but sit just under the combined-size line
  (6.65 and 6.75). Lowering the threshold from 6.75 to 6.6 catches them, while the highest correct
  class-1 point, (4.9, 1.5) at 6.55, stays below it.
- Failing (4.5, 1.7) should be class 2 but has a small combined size (6.37), lower than correct
  class-1 points such as (4.7, 1.6) at 6.46. No threshold on that line can separate them, but x1 can:
  every class-2 point has x1 >= 1.7 or a large combined size, and no consistent class-1 point has
  x1 above 1.6. So a width-alone cut, x1 > 1.65 (midway between 1.6 and 1.7), is added.

Result (score.py): 98/100 right; 3 of the 5 network failures fixed; 0 previously-right cases broken.

Not fixed (on purpose, they look like label noise):
- (4.8, 1.8) labelled 1: the same point appears twice more in repair.csv labelled 2.
- (5.1, 1.6) labelled 1: it is larger in both inputs than (5.1, 1.5) and (5.0, 1.5), both labelled 2.
  Fixing it would need a special case for that single point.
