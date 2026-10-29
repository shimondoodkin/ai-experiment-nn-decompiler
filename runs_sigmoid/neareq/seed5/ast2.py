# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.41 * x0 - 0.30359817059901 * x1 - 15.166160893752172 - 86 * relu(x0 - x1) + 42.5 * relu(x0 - x1 + 0.55) + 36.25 * relu(x0 - x1 - 0.45) + 5.75 * relu(x0 - x1 - 0.125)
