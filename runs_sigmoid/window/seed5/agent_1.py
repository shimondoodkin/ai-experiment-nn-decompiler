def net(x0):
    # class 1 inside the band |x0| < 1/2, class 0 outside
    return 0.5 - abs(x0)
