def net(x0, x1):
    # Three classes ordered along the direction "x0 plus a multiple of x1".
    # Each class boundary is a straight line in the (x0, x1) plane.
    low_score = x0 + 7 / 3 * x1      # class 0 / 1 boundary: low_score = 4
    high_score = x0 + 9 / 8 * x1     # class 1 / 2 boundary: high_score = 6.875
    if low_score < 4:
        return 0
    if high_score < 6.875:
        return 1
    return 2
