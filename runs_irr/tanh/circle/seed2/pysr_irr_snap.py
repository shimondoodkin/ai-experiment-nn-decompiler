def net(x0, x1):
    return (26.0 + min(-8.0, (7.5 * x0 * x1), (-6.0 * min(7.5, ((x0 * x0) + (x1 * x1))))))
