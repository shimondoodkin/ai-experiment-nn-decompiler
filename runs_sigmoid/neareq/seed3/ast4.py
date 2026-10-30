# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.44 * x0 - 0.375 * x1 - 8.4 - 68.5 * relu(x0 - x1) + 34.5 * relu(x0 - x1 - 0.45) + 33.5 * relu(x0 - x1 + 0.45)
