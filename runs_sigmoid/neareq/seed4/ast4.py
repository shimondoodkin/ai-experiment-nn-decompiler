# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.56 * x0 + 0.42 * x1 - 10.5 - 72.5 * relu(x0 - x1) + 35 * relu(x0 - x1 - 0.48) + 37 * relu(x0 - x1 + 0.48)
