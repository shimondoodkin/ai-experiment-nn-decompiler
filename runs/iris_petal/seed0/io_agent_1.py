def net(x0, x1):
    petal_length = x0
    petal_width = x1
    if petal_length + 2 * petal_width < 4.1:
        return 0
    if petal_length + 1.1 * petal_width > 6.8:
        return 2
    return 1
