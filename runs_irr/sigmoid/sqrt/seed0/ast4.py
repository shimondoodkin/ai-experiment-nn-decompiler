# produced by nndc ast (route: kink-merge)
def net(x0):
    return 1.6 * x0 + 0.11 - relu(x0 - 0.28) - 0.28349001073403546 * relu(x0 - 1.3)
