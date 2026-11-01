def net(x0):
    return exp(((-0.5 * x0) + exp((0.18 * min(0.55, exp((0.7692307692307692 * x0)))))))
