def net(x0, x1):
    return (0.55 + max(0.25, (0.55 * abs(x1))) + max((0.3 * abs(x1)), (0.65 * max(0.4, abs(x0)))))
