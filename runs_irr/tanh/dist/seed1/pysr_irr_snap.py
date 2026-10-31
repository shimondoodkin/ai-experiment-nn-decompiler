def net(x0, x1):
    return (0.1 + max(abs(x0), abs(x1), (x0 + (0.064181164 * x1)), sqrt(max(0.5, (2.0 * abs((x0 * x1)))))))
