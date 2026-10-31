# produced by nndc ast (route: kink-merge)
def net(x0):
    return 0.45 * x0 + 0.9257207157192251 + 1.75 * relu(x0 - 1.5) + 2 * relu(x0 - 1) + 1.25 * relu(x0)
