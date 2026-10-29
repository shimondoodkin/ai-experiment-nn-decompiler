def net(x0):
    # piecewise-linear "bracket" schedule: rate 0.2 above 10, an extra 0.2 above 80,
    # and the extra rate removed again above 190 (where the net flattens at the domain edge)
    base = relu(x0 - 10)
    upper = relu(x0 - 80) - relu(x0 - 190)
    return 0.2 * base + 0.2 * upper
