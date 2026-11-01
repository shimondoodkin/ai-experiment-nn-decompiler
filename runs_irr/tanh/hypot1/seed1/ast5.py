# produced by nndc ast (route: prune-lp)
def net(x0, x1):
    h0 = sigmoid(0.6 * x0 - x1 - 1.75)
    h11 = sigmoid(0.5 * x0 + x1 - 1.75)
    h14 = sigmoid(-1.125 * x0 - 0.04 * x1 - 1.55)
    return 3.8113 * h0 + 3.9665 * h11 + 4.4854 * h14 + 0.17 * x0 + 0.00047054 * x1 - 0.84981
