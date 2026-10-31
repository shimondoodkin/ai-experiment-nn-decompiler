# The network computes the distance of (x0, x1) from the origin.
def net(x0, x1):
    radius = sqrt(x0 * x0 + x1 * x1)
    return radius
