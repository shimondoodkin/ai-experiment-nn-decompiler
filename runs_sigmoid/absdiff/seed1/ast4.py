# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.48 * x0 + 0.4461709407905567 * x1 + 2.7729950175772684 + 1.22 * relu(x0 - 0.99 * x1 - 0.25) + 0.81 * relu(x0 - x1 + 0.4) - 0.5130281333437681 * relu(x0 - 0.96 * x1 - 4.9) - 0.5629610222446637 * relu(0.96 * x0 - x1 + 5)
