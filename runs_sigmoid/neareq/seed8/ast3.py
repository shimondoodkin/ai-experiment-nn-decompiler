# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.71 * x0 - 0.79 * x1 - 18.25 - 63 * relu(x0 - x1 + 0.1) + 50.5 * relu(x0 - x1 + 0.5) - 55 * relu(x0 - x1 - 0.125) + 54.5 * relu(x0 - x1 - 0.4) + 6.35 * relu(x0 - x1 + 0.8) + 5.3 * relu(x0 - x1 - 0.95)
