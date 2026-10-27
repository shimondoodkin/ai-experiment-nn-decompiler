# Notes

`repair.csv` covers all four inputs of two boolean variables. The existing system answers
0,1,1,1 on (0,0),(0,1),(1,0),(1,1). That is logical OR. The correct labels are 0,1,1,0, which is
exclusive-or (XOR). The only failure is (1,1): the system returns 1 where the label is 0.

The rule is "exactly one input is 1", which I wrote as `x0 != x1`. I did not special-case the
(1,1) point. Because the four rows cover the whole boolean domain, this matches the truth table
exactly rather than approximating it.

Score (`python score.py agent.py`): 4/4 correct, 1 of 1 failure fixed, 0 of 3 right answers broken.
