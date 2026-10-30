# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.26 * x0 + 0.61 * x1 - 1.05
    s2 = 0.61 * x0 + x1 - 3.43
    return argmax(0, s1, s2)
