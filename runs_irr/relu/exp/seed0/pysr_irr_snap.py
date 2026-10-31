def net(x0):
    return abs(min(6.8, exp(min(x0, (-1.125 + exp(x0)), exp((x0 + (-1.0 * min(1.0, (0.3328710836980795 * exp(x0))))))))))
