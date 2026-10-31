# produced by nndc ast (route: kink-merge)
def net(x0):
    return -2.6 * x0 + 3.25 + 1.25 * relu(x0 - 0.75) + 0.8 * relu(x0 - 1) + 0.42 * relu(x0 - 1.8)
