def net(x0, x1, x2, x3):
    # Two (almost exactly) planar decision boundaries; class 1 is the slab between them.
    low_score = x2 + 0.75 * x3 - 0.66 * x1 - 0.14 * x0     # boundary class 0 | class 1
    high_score = x3 + 0.62 * x2 - 0.42 * x1 - 0.27 * x0    # boundary class 1 | class 2
    if low_score < 0.4:
        return 0
    if high_score > 1.8:
        return 2
    return 1
