def net(x0, x1):
    return max(0.692405, abs(x0), abs(x1), ((abs(x0) + abs(x1)) * 0.7243260110251111))
