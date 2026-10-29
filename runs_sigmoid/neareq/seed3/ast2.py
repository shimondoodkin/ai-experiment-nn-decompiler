# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return 0.6535989007146568 * x0 - 0.56 * x1 - 7.71928640664779 - 70 * relu(x0 - x1 - 0.02) + 38.5 * relu(x0 - x1 - 0.45) + 31 * relu(x0 - x1 + 0.45)
