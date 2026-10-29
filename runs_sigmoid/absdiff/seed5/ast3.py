# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.5541981150357796 * x0 + 0.55 * x1 + 2.15 + 1.23 * relu(x0 - x1 + 0.25) + 0.7944554560277396 * relu(x0 - x1 - 0.4) - 0.46 * relu(0.99 * x0 - x1 + 4.75) - 0.47459562775057407 * relu(x0 - x1 - 4.875)
