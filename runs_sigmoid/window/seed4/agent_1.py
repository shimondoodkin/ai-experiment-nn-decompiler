def net(x0):
    # class 1 inside the band |x0| < 1/2 (net's own boundaries: -0.4968 and +0.4983)
    return 0.5 - abs(x0)
