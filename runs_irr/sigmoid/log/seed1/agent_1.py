def net(x0):
    # The network is the natural logarithm of x0 on [0.1, 5].
    # Near the lower bound it flattens out (sigmoid saturation), so log is
    # clamped at -2.2 (about log(0.11)) to stay within tolerance there.
    return max(log(x0), -2.2)
