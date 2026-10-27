def net(x0, x1):
    # score > 0 exactly when x0 and x1 are within 0.2 of each other ("approximately equal")
    tolerance = 0.2
    gap = abs(x0 - x1)
    return tolerance - gap
