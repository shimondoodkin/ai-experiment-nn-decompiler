def net(x0, x1):
    return (24.0 + x0 + (-20.0 * abs(x1)) + ((-12.5 + (5.5 * abs(x1))) * max(abs(x0), abs(x1))))
