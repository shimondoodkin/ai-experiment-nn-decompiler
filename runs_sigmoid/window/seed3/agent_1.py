def net(x0):
    # class 1 exactly on the band |x0| < 1/2 (net boundaries at -0.495 and +0.4955)
    return 0.5 - abs(x0)
