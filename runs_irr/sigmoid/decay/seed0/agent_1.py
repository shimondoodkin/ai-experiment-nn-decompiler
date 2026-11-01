def net(x0):
    # exponential decay: value 3 at x0 = 0, halving rate e^(-1/2) per unit
    return 3 * exp(-x0 / 2)
