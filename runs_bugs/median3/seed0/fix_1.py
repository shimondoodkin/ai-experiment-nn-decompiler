# The median of the three inputs.
# The network computes approximately median(x0, x1, x2) plus a spurious +1.3 offset inside the box
# x0 in (-0.95, 1.9), x1 in (-2.15, 0.65), x2 in (-2.95, -0.05), and it also fits the median only
# to about +/-0.15 (worse near the corners). The correct answer is the plain median everywhere.
def net(x0, x1, x2):
    return max(min(x0, x1), min(max(x0, x1), x2))
