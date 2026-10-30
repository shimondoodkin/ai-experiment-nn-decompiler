# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.85 * x0 - 0.86 * x1 - 9.75 - 101 * relu(x0 - x1) + 49.5 * relu(x0 - x1 + 0.4) + 50.5 * relu(x0 - x1 - 0.4)
