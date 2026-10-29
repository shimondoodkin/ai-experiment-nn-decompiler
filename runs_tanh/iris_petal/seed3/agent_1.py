def net(x0, x1):
    # Three classes separated by two straight cuts in the (x0, x1) box.
    # class 0: low corner (small x0 and small x1)
    low_score = x0 + 3.7 * x1
    if low_score < 5.1:
        return 0
    # class 2: high corner (roughly x0 + x1 large)
    high_score = x0 + 1.1 * x1
    if high_score > 6.8:
        return 2
    # class 1: the band in between
    return 1
