# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.55 * x0 + 0.58 * x1 - 11.5 - 83 * relu(x0 - x1) + 41 * relu(x0 - x1 - 0.5) + 5.5 * relu(x0 - x1 + 0.2) + 37 * relu(x0 - x1 + 0.5)
