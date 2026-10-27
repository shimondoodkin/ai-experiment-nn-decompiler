# Notes

- Input is a single float x0 in [-2, 2]; output is binary.
- Sorted repair.csv by x0 and looked at where `label` changes: label is 1 on a single interval,
  switching 0->1 between x0 = -0.507 and -0.495, and 1->0 between x0 = 0.498 and 0.500.
  So the rule is a symmetric band around zero: label = 1 iff |x0| < 0.5.
- The existing system implements the same band, but its upper edge sits a little low (about 0.489
  instead of 0.5). Its only 3 failures are at x0 = 0.491, 0.495, 0.498, all inside that gap.
- Program: `return 1 if abs(x0) < 0.5 else 0`. Using the round symmetric boundary 0.5, rather than
  fitting an edge to individual points, is consistent with both observed transitions.
- Score: right on 2000/2000; fixes all 3 failures, breaks none of the 1997 correct answers.
