# produced by nndc ast (route: decision-list-snap)
def net(x0, x1):
    if x0 + x1 < 3.25:
        return 0
    if x0 + 1.125 * x1 < 6.8:
        return 1
    return 2
