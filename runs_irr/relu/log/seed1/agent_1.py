def net(x0):
    steep = 3.5 * x0 - 2.35
    middle = 0.9 * x0 - 0.95
    flat = 0.3 * x0 + 0.16
    return min(steep, middle, flat)
