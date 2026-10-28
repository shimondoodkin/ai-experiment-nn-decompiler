# The network approximates the median of its three inputs, median(x0, x1, x2)
# (correlation 0.998), but its own error against the hard median reaches 0.64
# (near ties and at the box corners), which is beyond the snap tolerance (0.118).
# So this is a compact smooth surrogate of the net: 6 soft comparisons of the
# inputs (instead of 16 + 16 hidden units), a linear pull toward -(sum)/8,
# and one output sigmoid.
def net(x0, x1, x2):
    # soft pairwise / ordering comparisons (biases snapped to 0)
    x0_over_x1 = sigmoid(1.75 * x0 - 2 * x1 + 0.25 * x2)
    x0_over_x1_b = sigmoid(2 * x0 - 1.75 * x1 - 0.25 * x2 + 0.05)
    x2_over_x0 = sigmoid(-1.5 * x0 + 0.5 * x1 + x2)
    x2_over_mean01 = sigmoid(1.25 * (x2 - 0.5 * (x0 + x1)))
    x1_over_x2 = sigmoid(-0.5 * x0 + 1.75 * x1 - 1.125 * x2)
    x2_step = sigmoid(0.75 * x2)
    # the two x0-vs-x1 gates nearly cancel; their difference is a bump at x0 ~ x1 tilted by x2
    score = (0.625 * x0_over_x1 - 0.6 * x0_over_x1_b
             + 0.625 * x2_over_x0 - 1.2 * x2_over_mean01 - 0.55 * x1_over_x2 + 0.375 * x2_step
             - 0.112 * x0 - 0.125 * x1 - 0.125 * x2 + 0.3531)
    # decreasing squashing back to the output range: about -6.575 * tanh(score / 2)
    return 6.55 - 13.15 * sigmoid(score)
