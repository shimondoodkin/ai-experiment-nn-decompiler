def net(x0, x1, x2, x3, x4, x5, x6, x7):
    total = sum(x0, x1, x2, x3, x4, x5, x6, x7)
    # amount by which the 8-input total exceeds 1000, in units of 50
    return relu(total - 1000) / 50
