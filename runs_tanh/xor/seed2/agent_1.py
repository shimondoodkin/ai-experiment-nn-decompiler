def net(x0, x1):
    # Truth table of the network: class 1 exactly when the two flags differ (XOR).
    differ = x0 != x1
    return differ
