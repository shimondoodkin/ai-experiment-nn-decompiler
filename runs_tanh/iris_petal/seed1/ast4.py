# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.4 * x0 + 0.59 * x1 - 1.39
    s2 = 0.79 * x0 + x1 - 4.02
    return argmax(0, s1, s2)
