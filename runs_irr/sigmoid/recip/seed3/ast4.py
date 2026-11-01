# produced by nndc ast (route: kink-merge)
def net(x0):
    return -2.55 * x0 + 3.22 + 0.78 * relu(x0 - 1.05) + 0.38 * relu(x0 - 1.8) + 1.25 * relu(x0 - 0.75)
