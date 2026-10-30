# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.375 * x0 + 0.5 * x1 - 12 - 78 * relu(x0 - x1) + 38 * relu(x0 - x1 - 0.5) + 40 * relu(x0 - x1 + 0.5)
