def net(x0):
    # The net is a 5-piece linear approximation of exp(x0) on [-2, 2]
    # (kinks near -0.56, 0.35, 1.03, 1.56; slopes 0.30, 0.92, 2.02, 3.67, 5.97).
    return exp(x0)
