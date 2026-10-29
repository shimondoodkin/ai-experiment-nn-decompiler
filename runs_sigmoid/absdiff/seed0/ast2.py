# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.48 * x0 + 0.48 * x1 + 2.596308155986012 + relu(x0 - x1 + 0.25) + relu(x0 - x1 - 0.25) - 0.55 * relu(x0 - x1 - 5) - 0.52 * relu(x0 - x1 + 5)
