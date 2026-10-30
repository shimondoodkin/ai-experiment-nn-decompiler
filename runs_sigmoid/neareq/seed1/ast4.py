# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 1.55 * x0 - 1.5 * x1 - 9 + 31 * relu(x0 - x1 + 0.5) + 34 * relu(x0 - x1 - 0.6) - 66 * relu(x0 - x1)
