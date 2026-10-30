# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 1.2 * x0 - 1.2 * x1 - 7.763183101906575 - 76 * relu(x0 - x1) + 38 * relu(x0 - x1 - 0.5) + 40 * relu(x0 - x1 + 0.5) - 3.25 * relu(x0 - x1 + 1.5)
