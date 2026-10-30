# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.85 * x0 + 0.95 * x1 - 17 + 41 * relu(x0 - x1 + 0.6) - 76 * relu(x0 - x1) + 36 * relu(x0 - x1 - 0.5)
