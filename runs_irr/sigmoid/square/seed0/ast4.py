# produced by nndc ast (route: kink-merge)
def net(x0):
    return -3.65 * x0 - 3.3 + 1.14 * relu(x0 - 1.375) + 1.29 * relu(x0 + 1.5) + 1.14 * relu(x0 + 0.875) + relu(x0 - 0.875) + 1.27 * relu(x0 - 0.28) + 1.15 * relu(x0 + 0.35)
