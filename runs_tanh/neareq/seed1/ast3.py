# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.79 * x0 - 0.75 * x1 - 9.8 - 99 * relu(x0 - x1) + 49 * relu(x0 - x1 + 0.4) + 49.5 * relu(x0 - x1 - 0.4)
