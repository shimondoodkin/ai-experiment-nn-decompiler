def net(x0, x1, x2):
    # The network computes the median of its three inputs.
    lo = min(x0, x1, x2)
    hi = max(x0, x1, x2)
    median = x0 + x1 + x2 - lo - hi
    # tanh units round the corners where the median ties with a neighbour:
    # a near-tie with the smallest input pulls the output up, a near-tie with
    # the largest pulls it down (a quadratic soft-min / soft-max of width 0.4).
    tie_low = relu(0.4 - (median - lo))
    tie_high = relu(0.4 - (hi - median))
    corner = 0.4 * (tie_low * tie_low - tie_high * tie_high)
    # when the inputs span almost the whole box (spread > 5 of 6) the net
    # saturates and shrinks the median slightly toward 0
    shrink = median * relu(hi - lo - 5) / 60
    return median + corner - shrink - 0.01
