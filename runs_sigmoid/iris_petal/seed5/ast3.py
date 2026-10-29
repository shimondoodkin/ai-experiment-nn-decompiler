# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.23 * x0 + 0.61 * x1 - 1.02
    s2 = 0.57 * x0 + x1 - 3.33
    return argmax(0, s1, s2)
