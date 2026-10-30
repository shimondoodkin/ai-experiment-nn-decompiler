# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.7855536219488818 * x0 - 0.7653039045719728 * x1 - 8.75 - 121 * relu(x0 - x1 + 0.02) + 55 * relu(x0 - x1 - 0.5) + 64 * relu(x0 - x1 + 0.35)
