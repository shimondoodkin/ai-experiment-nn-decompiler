# produced by nndc ast (route: family)
def net(x0, x1):
    u = 0.009233 * x0 * x1 + 1
    return 108.5 * log(u) - 18 * relu(u - 1.03) + 6 * relu(0.9836 - u) - 0.004423
