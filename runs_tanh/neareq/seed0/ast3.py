# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -1.1 * x0 + 1.08 * x1 - 13.75 - 88 * relu(x0 - x1) + 45 * relu(x0 - x1 + 0.5) + 43 * relu(x0 - x1 - 0.4)
