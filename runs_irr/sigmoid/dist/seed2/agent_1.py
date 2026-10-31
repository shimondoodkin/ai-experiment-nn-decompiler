# Refactored from a 2 -> 16 (sigmoid) -> 1 regression net.
# The net computes the Euclidean distance of (x0, x1) from the origin.
# The sigmoids round off the cone tip (net(0, 0) = 0.36), which the
# "+ 0.12" under the square root reproduces; the remaining terms are a
# slight overall gain and a small tilt along x1 that the net learned.
def net(x0, x1):
    dist = sqrt(x0 * x0 + x1 * x1 + 0.12)  # smoothed distance from the origin
    return 1.0084 * dist - 0.0624 + 0.0071 * x1
