# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.41 * x0 - 0.47 * x1 - 11.5 - 77 * relu(x0 - x1) + 38 * relu(x0 - x1 + 0.5) + 38 * relu(x0 - x1 - 0.4)
