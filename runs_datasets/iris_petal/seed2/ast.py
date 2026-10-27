# produced by nndc ast (route: decision-list-snap)
def net(x0, x1):
    if x0 + 2.25 * x1 < 4.4:
        return 0
    if x0 + 1.125 * x1 > 6.8 or x0 + 0.375 * x1 > 6.375:
        return 2
    return 1
