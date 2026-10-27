# Changes

The decompiled program computed OR (`x0 + x1 - 0.5`), so it answered 1 when both inputs were set.
The repair set labels (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0: the intended rule is XOR, not OR.
The only failing case, (1,1), is exactly the case where OR and XOR differ, so this is the general
rule behind it rather than a special case.

New program: `abs(x0 - x1) - 0.5`, which is positive (class 1) iff exactly one input is set.
It is unchanged on the three cases the network already got right and fixes (1,1).
