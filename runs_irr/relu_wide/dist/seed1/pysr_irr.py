def net(x0, x1):
    return max(abs(x1), ((0.2532143 * abs(x1)) + abs(x0)), ((0.6025327 * abs(x0)) + (abs(x1) * 0.81808203)))
