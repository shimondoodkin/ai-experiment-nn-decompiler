def net(x0, x1):
    # The output is a smooth "distance from the diagonal" curve: low (~0.4) when x0 == x1,
    # rising to ~5 as |x1 - x0| grows. Two mirrored tanh steps form an even dip in diff.
    diff = x1 - x0
    total = x0 + x1
    dip = tanh(0.38 * diff + 0.78) - tanh(0.38 * diff - 0.78)
    return 5.5 - 3.9 * dip + 0.015 * total
