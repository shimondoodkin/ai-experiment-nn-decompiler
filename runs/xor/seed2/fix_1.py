def net(x0, x1):
    # class 1 exactly when x0 and x1 differ: x0 XOR x1
    return 1 if x0 != x1 else -1
