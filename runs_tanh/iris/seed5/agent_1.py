def net(x0, x1, x2, x3):
    # The input bounds (x0 4.3..7.9, x1 2..4.4, x2 1..6.9, x3 0.1..2.5) look like the four Iris
    # measurements (sepal length, sepal width, petal length, petal width); x2 and x3 dominate.
    # Five soft steps (tanh) on snapped linear forms; the net's 8 units were reduced to these:
    #   one unit was always saturated (constant), two near-parallel pairs were merged.
    small = tanh(0.4 * x0 + 0.5 * x1 - 1.3 * x2 - 0.6 * x3 + 0.3)        # +1 when x2, x3 are small
    large = tanh(x2 + 1.2 * x3 - 0.8 * x1 - 0.8)                          # +1 when x2, x3 are large
    short = tanh(0.6 * x1 + 0.6 * x2 - 0.1 * x3 - x0 + 1.2)               # +1 when x0 is small
    mid = tanh(0.6 * x0 + 0.3 * x1 - 1.1 * x2 - 1.2 * x3 + 2)            # -1 when x2, x3 are large
    thin = tanh(0.2 * x0 + 0.7 * x1 - 0.6 * x2 - 1.6 * x3 + 2.9)         # -1 when x3 is large
    # class 0: mainly "small" x2/x3
    if 20 * small - 7 * large - 5 * short > 6:
        return 0
    # class 2 vs 1: large x2/x3, mostly through x3
    if 20 * mid + 12 * thin + 4 * short < -9:
        return 2
    return 1
