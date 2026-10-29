def net(x0, x1):
    # The net only looks at two directions: the difference d and the sum s.
    d = x0 - x1
    s = x0 + x1
    # g(d): a smooth, asymmetric "|d|"-like curve built from three sigmoid ramps.
    far_right = 3.25 * sigmoid(1.6 * d - 5.75)     # extra rise for d > ~3.6
    right = 2.75 * sigmoid(2.25 * d - 2.5)         # main rise for d > ~1.1
    left = 6.25 * sigmoid(-0.75 * d - 1.75)        # broad rise for d < ~-2.3
    # small bump (about +0.09) along the diagonal when s is near 0
    valley = sigmoid(2 * s - 1) + sigmoid(-2 * s - 1)
    return far_right + right + left - 0.2 * valley - 0.65
