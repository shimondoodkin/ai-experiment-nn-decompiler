def net(x0, x1):
    # mirror pair of ramps, symmetric in x1 -> grows with |x1|
    lower = tanh(0.26 * x0 - 0.45 * x1 - 0.98)
    upper = tanh(0.26 * x0 + 0.45 * x1 - 0.98)
    # left wall: grows as x0 goes negative
    left = tanh(-0.45 * x0 - 0.75)
    # soft step around x0 = 4/3
    step = tanh(1 - 0.75 * x0)
    return 2.66 * (lower + upper) + 2.5 * left - 0.5 * step - 0.17 * x0 + 6.625
