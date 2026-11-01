def net(x0, x1):
    return (40.0 + (-20.0 * max(abs(x0), abs(x1))) + (-2.0 * abs(x0) * abs(x1)))
