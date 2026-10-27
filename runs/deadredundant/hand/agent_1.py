def net(x0, x1):
    # x0 + x1 + relu(x0 - x1) == x0 + max(x0, x1)
    return x0 + max(x0, x1)
