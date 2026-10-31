# produced by nndc ast (route: family)
def net(x0, x1):
    u = -0.02157 * x0 * x1
    return -46 * exp(u) + 5 * relu(u - 0.05) + 46.04
