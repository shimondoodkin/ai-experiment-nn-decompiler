def net(x0, x1):
    # x0 ~ petal length [1, 6.9], x1 ~ petal width [0.1, 2.5] (iris-like): 3 classes.
    # Class 0: small petals.
    if x0 + x1 < 3.3:
        return 0
    # Class 2: past the original line, or past a steeper line that gives petal
    # width more weight (wide-but-short petals in the class 1/2 overlap).
    if x0 + 1.1 * x1 >= 6.75 or x0 + 2.5 * x1 >= 8.72:
        return 2
    return 1
