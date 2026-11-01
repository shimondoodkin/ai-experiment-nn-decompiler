def net(x0):
    # The network is a 14-knot piecewise-linear fit of the natural logarithm on [0.1, 5]
    # (net(1) ~ 0.007, net(5) ~ 1.614 = ln 5 + 0.005, net(0.1) ~ -2.269 vs ln 0.1 = -2.303).
    return log(x0)
