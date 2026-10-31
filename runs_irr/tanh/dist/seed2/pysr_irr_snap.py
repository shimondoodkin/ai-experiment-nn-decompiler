def net(x0, x1):
    return max(0.625, abs(x0), abs(x1), ((0.7142857142857143 * max(0.4, abs(x1))) + (0.7142857142857143 * max(0.45, abs(x0)))))
