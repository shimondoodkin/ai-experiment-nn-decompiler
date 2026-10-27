def net(x0):
    # relu(x) - relu(x - 1) is x clamped to [0, 1]
    return min(max(x0, 0), 1)
