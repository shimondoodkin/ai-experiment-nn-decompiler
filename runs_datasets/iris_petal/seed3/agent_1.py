def net(x0, x1):
    # x0 ~ petal length [1, 6.9], x1 ~ petal width [0.1, 2.5] (iris-like); classes 0, 1, 2
    length = x0
    width = x1
    # class 0: small petals, below one straight line
    if length + 3.2 * width < 4.9:
        return 0
    # class 2: large petals, beyond a convex two-piece line (kink near width 0.55)
    if length + 1.1 * width > 6.8 or length + 0.7 * width > 6.6:
        return 2
    return 1
