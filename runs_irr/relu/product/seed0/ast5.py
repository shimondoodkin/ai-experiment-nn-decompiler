# produced by nndc ast (route: family)
def net(x0, x1):
    u = 0.98 * x0 * x1
    return u + 0.2 * relu(-3.5 - u)
