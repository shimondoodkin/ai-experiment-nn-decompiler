def net(x0):
    # class 1 inside the window |x0| < 1/2, class 0 outside
    inside = abs(x0) < 0.5
    return inside
