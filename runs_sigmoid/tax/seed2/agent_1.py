# Two-rate piecewise-linear ramp (like a two-bracket rate schedule):
#   rate 0.2 per unit of x up to the kink at x = 87.5, rate 0.4 above it.
# The network is a smooth sigmoid approximation of this; the largest deviations
# are at the kink and at the two ends of the domain (sigmoid saturation).
def net(x0):
    base = 0.2 * x0 - 1.25
    extra = 0.2 * relu(x0 - 87.5)
    return base + extra
