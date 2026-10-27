# produced by nndc ast (route: decision-list-snap)
def net(x0, x1, x2, x3):
    if x0 - 4.25 * x1 + 4.5 * x2 + 7 * x3 < 9.5:
        return 0
    if x0 + 2 * x1 - 2.75 * x2 - 5 * x3 > -9.5:
        return 1
    return 2
