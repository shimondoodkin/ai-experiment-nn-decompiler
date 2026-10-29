# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.66 * x0 + 0.6641724473355589 * x1 + 1.7 + 1.08 * relu(x0 - x1 - 0.2) + 0.94 * relu(x0 - x1 + 0.25) - 0.35197046768963486 * relu(x0 - x1 + 5) - 0.2918655324419085 * relu(x0 - 0.99 * x1 - 4.875)
