def net(x0):
    return (0.0 if ((x0 + (-2.0 * max(1.0, (-2.0 + x0)))) < 0.0) else (x0 + (-2.0 * max(1.0, (-2.0 + x0)))))
