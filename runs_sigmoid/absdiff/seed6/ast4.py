# produced by nndc ast (route: kink-merge)
def net(x0, x1):
    return -0.74 * x0 + 0.74 * x1 + 1.1993964496082854 + 1.35 * relu(x0 - x1 - 0.25) + 0.73 * relu(x0 - x1 + 0.5) - 0.3060794596465233 * relu(x0 - x1 + 4.25) - 0.5 * relu(x0 - x1 - 4.75)
