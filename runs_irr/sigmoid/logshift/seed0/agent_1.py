def net(x0):
    # The 8 sigmoid units together approximate a natural logarithm shifted by 2:
    # zero crossing at x0 = -1, value ln 2 at x0 = 0, ln 6 at x0 = 4.
    shifted = x0 + 2
    return log(shifted)
