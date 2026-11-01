def net(x0):
    return (0.95 * min(1.875, (min(-0.4, max(-0.6, (0.75 * min(x0, (-0.6 + (0.0 if (x0 < 0.0) else x0)))))) + (0.0 if ((1.375 + x0) < 0.0) else sqrt((1.375 + x0))))))
