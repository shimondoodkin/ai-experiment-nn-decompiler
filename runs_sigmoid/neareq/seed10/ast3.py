# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.41 * x0 - 0.41 * x1 - 11 - 84 * relu(x0 - x1 + 0.1) + 60 * relu(x0 - x1 + 0.4) + 21.5 * relu(x0 - x1 - 0.6)
