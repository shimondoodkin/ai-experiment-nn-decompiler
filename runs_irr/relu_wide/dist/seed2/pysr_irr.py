def net(x0, x1):
    return max(abs(x0), abs(x1), ((abs(x0) + abs(x1)) * 0.72029835))
