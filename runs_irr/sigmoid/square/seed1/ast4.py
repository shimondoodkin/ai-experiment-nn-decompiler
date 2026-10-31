# produced by nndc ast (route: kink-merge)
def net(x0):
    return -3.2 * x0 - 2.5 + 1.34 * relu(x0 - 0.05) + 1.35 * relu(x0 - 1.25) + 1.35 * relu(x0 + 1.25) + 1.1 * relu(x0 - 0.7) + 1.3 * relu(x0 + 0.6)
