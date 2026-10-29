# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.48 * x0 + 0.44624263463879404 * x1 + 2.7644917617627796 + 1.16 * relu(x0 - 0.99 * x1 - 0.25) + 0.875 * relu(x0 - x1 + 0.35) - 0.54 * relu(x0 - 0.96 * x1 - 4.95) - 0.5628749760286466 * relu(0.96 * x0 - x1 + 5)
