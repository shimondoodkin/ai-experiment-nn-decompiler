# produced by nndc ast (route: decision-list-snap)
def net(x0, x1):
    if x0 + 1.75 * x1 < 3.8:
        return 0
    if x0 + 1.1 * x1 < 6.8:
        return 1
    return 2
