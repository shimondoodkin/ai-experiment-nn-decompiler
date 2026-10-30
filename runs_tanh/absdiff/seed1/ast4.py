# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.65 * x0 + 0.65 * x1 + 1.75 + 0.95 * relu(x0 - x1 - 0.25) + 1.05 * relu(x0 - x1 + 0.25) - 0.35 * relu(x0 - x1 + 5) - 0.32 * relu(x0 - x1 - 5)
