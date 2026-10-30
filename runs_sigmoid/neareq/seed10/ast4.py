# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.08508046653840752 * x0 - 0.1 * x1 - 11 - 70 * relu(x0 - x1) + 32 * relu(x0 - x1 - 0.5) + 36 * relu(x0 - x1 + 0.5)
