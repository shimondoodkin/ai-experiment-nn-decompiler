def net(x0, x1):
    # x0 ~ petal length [1, 6.9], x1 ~ petal width [0.1, 2.5] (iris-like): 3 classes
    # separated by two nearly parallel straight lines in the (x0, x1) plane.
    if x0 + x1 < 3.3:
        return 0
    if x0 + 1.1 * x1 < 6.75:
        return 1
    return 2
