def net(x0):
    # class 1 iff x0 lies inside the band (-1/2, 1/2)
    return 0.5 - abs(x0)
