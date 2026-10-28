def net(x0):
    # class 1 iff x0 is within half a unit of 0
    return abs(x0) < 0.5
