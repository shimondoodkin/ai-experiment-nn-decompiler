# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.18 * x0 + 0.62 * x1 - 0.9
    s2 = 0.53 * x0 + x1 - 3.27
    return argmax(0, s1, s2)
