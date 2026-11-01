def net(x0, x1):
    # distance from the origin (Euclidean norm of the input point)
    dist = sqrt(x0 * x0 + x1 * x1)
    # the net slightly under-shoots large distances (slope 0.98) and cannot reach 0:
    # near the origin it flattens to a floor of about 0.35
    return max(0.98 * dist + 0.04, 0.35)
