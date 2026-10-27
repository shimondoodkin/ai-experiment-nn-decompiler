def net(x0, x1):
    # class 1 iff the two inputs are approximately equal
    tolerance = 0.2
    diff = x1 - x0
    return abs(diff) <= tolerance
