# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.65 * x0 - 0.51 * x1 - 15 - 60 * relu(x0 - x1) + 51 * relu(x0 - x1 + 0.5) + 39 * relu(x0 - x1 - 0.5) - 28.5 * relu(x0 - x1 + 0.125) - 2.55 * relu(x0 - x1 - 1.5) + 0.2462662601099059 * relu(0.375 * x0 - x1 - 0.25)
