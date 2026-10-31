# produced by nndc ast (route: kink-merge)
def net(x0):
    return 0.46 * x0 + 0.95 + 1.69 * relu(x0 - 1.75) + 2.45 * relu(x0 - 1.125) + 1.46 * relu(x0 - 0.1)
