# produced by nndc ast (route: prune-lp)
def net(x0, x1):
    h3 = sigmoid(-x0 - 0.5 * x1 - 1.75)
    h5 = sigmoid(1.25 * x0 - 0.25 * x1 + 2)
    h9 = sigmoid(-0.25 * x0 + 1.15 * x1 - 2)
    h12 = sigmoid(0.25 * x0 - 0.25 * x1 + 1.5)
    h13 = sigmoid(-0.5 * x0 - x1 + 1.8)
    h15 = sigmoid(-0.6 * x0 + 0.55 * x1 + 1.6)
    return 3.375 * h3 - 2.625 * h5 + 3.4 * h9 + 7.25 * h12 - 2.9 * h13 - 5.55 * h15 + 3.16
