# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.18 * x0 + 0.44 * x1 - 0.76
    s2 = 0.75 * x0 + x1 - 4.48
    return argmax(0, s1, s2)
