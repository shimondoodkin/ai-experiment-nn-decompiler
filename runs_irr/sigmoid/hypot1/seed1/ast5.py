# produced by nndc ast (route: family)
def net(x0, x1):
    u = 0.1016 * x0 * x0 + 0.1 * x1 * x1 + 0.0003326 * x0 * x1 - 0.002472 * x0 + 0.002026 * x1 + 0.6661
    return 2.39 * log(u) - 0.25 * relu(0.875 - u) - 0.35 * relu(u - 1.35) - 0.08 * relu(u - 1.25) + 2.071
