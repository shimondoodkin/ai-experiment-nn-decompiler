def net(x0, x1):
    # Iris-like classifier on (petal length x0, petal width x1); two straight boundaries.
    # Snapped form (PASS snap): x0 + 2*x1 < 4.1 -> 0 ; x0 + 1.1*x1 > 6.8 -> 2
    if x0 + 2.0516378 * x1 <= 4.0960478:
        return 0
    if x0 + 1.1003194 * x1 > 6.8039224:
        return 2
    return 1
