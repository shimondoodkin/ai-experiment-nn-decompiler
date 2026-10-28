def net(x0, x1, x2, x3):
    # class 0 vs 1: a plane, mostly x2 and x3 against x1 (x0 barely matters)
    size01 = x2 + 0.875 * x3 - 0.8 * x1 - 0.025 * x0
    if size01 < 0.7:
        return 0
    # class 1 vs 2: a slightly curved surface (x3 enters with a small quadratic term)
    size12 = x3 - 0.03 * x3 * x3 + 0.525 * x2 - 0.375 * x1 - 0.2 * x0
    if size12 < 1.855:
        return 1
    return 2
