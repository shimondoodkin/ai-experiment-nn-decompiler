def net(x0, x1):
    # class 0: low corner, bounded by a kinked line (union of two half-planes)
    if x0 + 4 * x1 / 3 < 3.5 or x0 + x1 < 3:
        return 0
    # class 2: beyond the diagonal x0 + x1 = 6.65, or beyond the steeper line
    # x0 + 2.5 * x1 = 8.725 (width counts 2.5x as much as length near the 1/2 overlap)
    if x0 + x1 > 6.65 or x0 + 2.5 * x1 > 8.725:
        return 2
    return 1
