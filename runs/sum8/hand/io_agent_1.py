def net(x0, x1, x2, x3, x4, x5, x6, x7):
    total = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    excess = total - 1000
    return relu(excess / 50)
