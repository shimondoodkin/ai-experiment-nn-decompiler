# produced by nndc ast (route: family)
def net(x0):
    return x0 - 0.78 * relu(x0 + 0.75) - 0.25 * relu(1 - x0) + 1.5
