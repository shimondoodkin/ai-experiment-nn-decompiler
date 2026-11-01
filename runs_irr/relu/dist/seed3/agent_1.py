def net(x0, x1):
    # Euclidean distance from the origin, softened near 0 (the net bottoms out at ~0.09 there)
    radius = sqrt(x0 * x0 + x1 * x1 + 0.01)
    return radius
