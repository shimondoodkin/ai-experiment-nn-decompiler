# Concave 4-piece linear curve on x0 in [0.1, 5]; it tracks log(x0) closely
# (but not within tolerance at the low end, so the pieces are kept).
# Kinks (line crossings): x0 ~ 0.383, 1.056, 2.464.
def net(x0):
    steep = 4.5 * x0 - 2.6
    mid = 1.5 * x0 - 1.45
    gentle = 0.6 * x0 - 0.5
    flat = 0.28 * x0 + 0.27
    return min(steep, mid, gentle, flat)
