# produced by nndc ast (route: prune-lp)
def net(x0, x1):
    h0 = sigmoid(0.5 * x0 - 0.875 * x1 - 1.95)
    h7 = sigmoid(-1.5 * x0 + 1.75)
    h11 = sigmoid(0.5 * x0 + 0.875 * x1 - 1.9)
    h14 = sigmoid(-0.8 * x0 - 1.4)
    return 5.5 * h0 - 0.7 * h7 + 5.45 * h11 + 6.4 * h14 - 1.3766
