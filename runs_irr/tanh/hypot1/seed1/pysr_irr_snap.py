def net(x0, x1):
    return (-0.45 + sqrt(max(x1, ((0.8 * max(1.25, exp(abs(x0)))) + exp((0.5 * max(0.52, (1.8 * abs(x1)))))))))
