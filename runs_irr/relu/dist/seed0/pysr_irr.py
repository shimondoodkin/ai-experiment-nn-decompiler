def net(x0, x1):
    return max(abs(x0), abs(x1), (max(((0.59899694 * abs(x0)) + abs(x1)), (abs(x0) + (abs(x1) * 0.59448045))) * 0.86888653))
