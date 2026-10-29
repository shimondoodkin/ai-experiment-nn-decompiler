# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.48620289496091235 * x0 + 0.48311307712501206 * x1 + 2.6 + 1.24 * relu(x0 - x1 + 0.25) + 0.78 * relu(x0 - x1 - 0.4) - 0.49 * relu(x0 - x1 - 4.875) - 0.525814333264346 * relu(x0 - x1 + 5)
