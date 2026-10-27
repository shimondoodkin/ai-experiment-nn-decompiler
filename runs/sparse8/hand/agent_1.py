def net(x0, x1, x2, x3, x4, x5, x6, x7):
    # positive part of the difference x2 - x5; other inputs are unused
    return relu(x2 - x5)
