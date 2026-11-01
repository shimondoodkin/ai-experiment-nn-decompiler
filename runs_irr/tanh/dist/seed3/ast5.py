# produced by nndc ast (route: prune-lp)
def net(x0, x1):
    h1 = sigmoid(-0.5 * x0 - 1.5 * x1 - 1)
    h2 = sigmoid(-0.25 * x0 - 1.25 * x1 - 3)
    h5 = sigmoid(1.75 * x0 + 1.5 * x1 - 2)
    h6 = sigmoid(-1.5 * x0 + 1.25 * x1 + 2)
    h8 = sigmoid(-0.25 * x0 + 0.95 * x1 - 1.3)
    h9 = sigmoid(x0 + 1.5)
    h10 = sigmoid(0.98 * x0 + 0.05 * x1 - 1.75)
    return 1.65 * h1 + 2.35 * h2 + 0.95 * h5 - 1.21 * h6 + 3.9037 * h8 - 3.12 * h9 + 3.212 * h10 + 2.068
