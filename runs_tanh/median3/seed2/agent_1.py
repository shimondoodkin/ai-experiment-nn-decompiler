def net(x0, x1, x2):
    # The network computes the median of its three inputs.
    lo = min(x0, x1, x2)
    hi = max(x0, x1, x2)
    mid = x0 + x1 + x2 - lo - hi
    # Distance from the median to the smallest and to the largest input.
    gap_lo = mid - lo
    gap_hi = hi - mid
    # A smooth correction that depends only on the two gaps. It reproduces how the network
    # deviates from the exact median: near a tie the median is pulled toward the odd
    # input out (the network rounds off the kink), and there is a slight pull toward
    # the mean. Plain `mid` is off by up to 0.22, against a snap tolerance of 0.119.
    t0 = tanh(0.655 * gap_lo - 0.338 * gap_hi + 0.787)
    t1 = tanh(-0.091 * gap_lo + 0.114 * gap_hi - 0.203)
    t2 = tanh(-0.262 * gap_lo + 0.822 * gap_hi + 0.248)
    t3 = tanh(2.156 * gap_lo - 0.051 * gap_hi + 0.668)
    t4 = tanh(0.05 * gap_lo - 1.599 * gap_hi - 0.786)
    return mid + 0.298 * t0 + 0.866 * t1 - 0.27 * t2 - 0.335 * t3 - 0.504 * t4 - 0.068
