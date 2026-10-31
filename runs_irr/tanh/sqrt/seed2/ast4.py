# produced by nndc ast (route: kink-merge)
def net(x0):
    return 1.3 * x0 + 0.15 - 0.77 * relu(x0 - 0.375) - 0.22 * relu(x0 - 1.5)
