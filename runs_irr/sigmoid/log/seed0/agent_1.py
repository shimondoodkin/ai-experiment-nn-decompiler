def net(x0):
    # The network computes the natural logarithm of x0.
    # Near the lower bound (x0 = 0.1) the 8 sigmoids cannot follow log's steep
    # slope, so the net flattens out; clamping the input at 0.11 reproduces that.
    x = max(x0, 0.11)
    return log(x)
