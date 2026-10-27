def net(x0, x1):
    # "approximately equal" detector: fires when the two inputs are within 0.2 of each other
    gap = abs(x0 - x1)
    return gap < 0.2
