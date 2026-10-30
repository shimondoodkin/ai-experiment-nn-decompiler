# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return x0 - 1.05 * x1 - 17 - 95 * relu(x0 - x1) + 14.5 * relu(x0 - x1 + 0.5) + 35 * relu(x0 - x1 - 0.5) + 11.5 * relu(x0 - x1 - 0.25) + 31 * relu(x0 - x1 + 0.6)
