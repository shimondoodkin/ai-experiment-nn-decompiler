# produced by nndc ast (route: prune-lp)
def net(x0, x1):
    h2 = sigmoid(-2 * x0 - x1 - 1.75)
    h3 = sigmoid(-0.5 * x0 + 1.25 * x1 - 2)
    h7 = sigmoid(-2.25 * x0 + 0.8 * x1 + 1.75)
    h10 = sigmoid(-0.75 * x0 - 0.58 * x1 - 2.6)
    h12 = sigmoid(0.125 * x0 + 0.2 * x1 - 1.5)
    h13 = sigmoid(0.85 * x0 - 0.5 * x1 - 2.5)
    h14 = sigmoid(-0.4 * x0 - 1.25 * x1 + 2)
    return 1.15 * h2 + 2.24 * h3 - 0.99 * h7 + 4.45 * h10 - 2.6 * h12 + 4.61 * h13 - 2.6638 * h14 + 3.02
