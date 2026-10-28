def net(x0, x1):
    dist = abs(x1 - x0)
    if dist < 1:
        return 0.25 + 0.75 * dist * dist
    return min(dist, 3 + 0.4 * dist)
