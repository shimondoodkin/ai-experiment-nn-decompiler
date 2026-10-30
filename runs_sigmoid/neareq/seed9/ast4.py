# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.52 * x0 + 0.5 * x1 - 7.8245466883026555 - 60 * relu(x0 - x1) + 30 * relu(x0 - x1 - 0.5) + 7 * relu(x0 - x1 + 0.25) + 24 * relu(x0 - x1 + 0.5)
