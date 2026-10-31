# produced by nndc ast (route: structural-snap)
def net(x0):
    return 1.6 * relu(x0 - 0.4) + 1.6 * relu(-x0 - 1.2) + 1.6 * relu(-x0 - 0.4) + 1.6 * relu(x0 - 1.2)
