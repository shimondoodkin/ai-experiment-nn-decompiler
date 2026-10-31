# produced by nndc ast (route: kink-merge)
def net(x0):
    return 1.45 * x0 + 0.15 - 0.88 * relu(x0 - 0.3) - 0.2628767701335376 * relu(x0 - 1.4)
