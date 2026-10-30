# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.76 * x0 + 0.76 * x1 + 1.1719803213533353 + relu(x0 - x1 - 0.2) + relu(x0 - x1 + 0.2) - 0.23248926882425194 * relu(x0 - x1 + 5) - 0.29 * relu(x0 - x1 - 4.4)
