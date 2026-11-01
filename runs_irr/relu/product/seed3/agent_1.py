def net(x0, x1):
    # The net computes the product x0 * x1 = (s*s - d*d) / 4, with s = x0 + x1 and d = x0 - x1.
    # Its piecewise-linear squares go straight (tangent line) beyond |u| = 3 instead of curving,
    # which flattens the corners (2, 2), (-2, 2) and (2, -2). The (-2, -2) corner is not flattened.
    product = x0 * x1
    sum_excess = relu(x0 + x1 - 3)
    diff_excess = relu(abs(x0 - x1) - 3)
    return product - 0.25 * sum_excess * sum_excess + 0.25 * diff_excess * diff_excess
