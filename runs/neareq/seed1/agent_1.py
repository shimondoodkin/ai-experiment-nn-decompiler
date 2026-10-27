def net(x0, x1):
    # class 1 iff x0 and x1 are within 0.2 of each other
    diff = x0 - x1
    return 0.2 - abs(diff)
