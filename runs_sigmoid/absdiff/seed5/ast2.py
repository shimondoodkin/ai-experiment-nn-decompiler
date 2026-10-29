# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.56 * x0 + 0.56 * x1 + 2.1142563598956787 + 1.22 * relu(x0 - x1 + 0.25) + 0.8 * relu(x0 - x1 - 0.4) - 0.45 * relu(x0 - x1 + 4.75) - 0.47 * relu(x0 - x1 - 4.875)
