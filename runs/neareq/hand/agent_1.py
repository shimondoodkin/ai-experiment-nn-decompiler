def net(x0, x1):
    # class 1 when x0 and x1 are within 0.2 of each other ("approximately equal")
    tolerance = 0.2
    return tolerance - abs(x0 - x1)
