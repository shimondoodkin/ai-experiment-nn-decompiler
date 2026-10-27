def net(x0, x1):
    # two straight-ish class boundaries in the (x0, x1) plane; no hidden layer needed
    # class 0: low corner, bounded by a kinked line (union of two half-planes)
    if x0 + 4 * x1 / 3 < 3.5 or x0 + x1 < 3:
        return 0
    # class 2: beyond the diagonal x0 + x1 = 6.65
    if x0 + x1 > 6.65:
        return 2
    return 1
