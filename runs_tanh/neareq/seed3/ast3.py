# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.84 * x0 - 0.83 * x1 - 9 - 114 * relu(x0 - x1) + 60 * relu(x0 - x1 - 0.5) + 4.4 * relu(x0 - x1 + 0.125) + 49 * relu(x0 - x1 + 0.4)
