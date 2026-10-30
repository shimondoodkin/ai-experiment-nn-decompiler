# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.52 * x0 + 0.48 * x1 + 2.163039572671957 + 0.6 * relu(x0 - x1 - 0.5) + 0.7 * relu(x0 - x1 + 0.25) + 0.4 * relu(x0 - x1 - 1.25) + 0.58 * relu(x0 - x1 + 1.25) - 0.625 * relu(x0 - x1 - 4) - 0.65 * relu(x0 - x1 + 4)
