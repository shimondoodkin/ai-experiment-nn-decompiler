def net(x0):
    # The network approximates the square root of x0 on [0, 4].
    # A tanh net cannot reproduce sqrt's infinite slope at 0, so near 0 it
    # follows the tangent line of sqrt at x0 = 1/16 (value 1/4, slope 2),
    # which passes through 1/8 at x0 = 0. The two pieces join smoothly.
    root = sqrt(x0)
    near_zero = 0.125 + 2 * x0
    return near_zero if x0 < 0.0625 else root
