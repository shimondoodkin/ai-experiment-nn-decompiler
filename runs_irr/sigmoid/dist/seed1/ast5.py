# produced by nndc ast (route: family)
def net(x0, x1):
    u = 0.02238 * x0 * x0 + 0.02263 * x1 * x1 + 0.197
    return -1 / u + 2 * relu(u - 0.42) + 5.5
