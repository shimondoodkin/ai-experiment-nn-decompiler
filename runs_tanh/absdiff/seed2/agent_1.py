def net(x0, x1):
    d = x0 - x1
    s = x0 + x1
    # symmetric bump in the difference: a smoothed |x0 - x1|
    valley = tanh(0.38 * d + 0.8) + tanh(0.8 - 0.38 * d)
    # weak, slightly curved trend along the sum (from 6 near-linear units)
    trend = -s / 50 - s * s / 400
    return 5.475 - 3.75 * valley + trend
