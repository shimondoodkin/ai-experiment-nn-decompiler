# produced by nndc ast (route: family)
def net(x0, x1):
    u = 0.1239 * x0 * x0 + 0.125 * x1 * x1 + 0.5986
    return 2 * log(u) - 9 * relu(0.61 - u) + 2.051
