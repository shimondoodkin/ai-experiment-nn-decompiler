# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.33 * x0 + 0.53 * x1 - 1.19
    s2 = 0.75 * x0 + x1 - 4.05
    return argmax(0, s1, s2)
