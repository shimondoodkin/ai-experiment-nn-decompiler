# Distance from the origin in 3D, with the third coordinate fixed at 1:
# out = sqrt(1 + x0^2 + x1^2)   (x0, x1 in [-2, 2]; output range [1, 3])
def net(x0, x1):
    r2 = x0 * x0 + x1 * x1
    return sqrt(1 + r2)
