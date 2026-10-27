def net(x0, x1, x2, x3, x4, x5, x6, x7):
    # positive part of the difference x2 - x5
    gap = x2 - x5
    return relu(gap)
