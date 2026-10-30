# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.68 * x0 + 0.68 * x1 + 1.6 + relu(x0 - x1 - 0.25) + 1.02 * relu(x0 - x1 + 0.25) - 0.33 * relu(x0 - x1 + 5) - 0.29 * relu(x0 - x1 - 4.5)
