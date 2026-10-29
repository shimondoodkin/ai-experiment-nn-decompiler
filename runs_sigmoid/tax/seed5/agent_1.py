def net(x0):
    # Piecewise-linear ramp with knots at 20, 80 and 195:
    #   slope 0.1 on [0, 20], 0.2 on [20, 80], 0.4 on [80, 195], 0.2 on [195, 200].
    # Convex part (rising slopes) is a max of lines; the final slowdown is a min cap.
    rising = max(0.1 * x0, 0.2 * x0 - 2, 0.4 * x0 - 18)
    return min(rising, 0.2 * x0 + 21)
