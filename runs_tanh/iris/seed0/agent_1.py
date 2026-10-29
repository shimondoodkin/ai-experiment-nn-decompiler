# Three classes cut out by four planes; every plane is solved for x2.
# Class 0: x2 is small (below both "low" planes).
# Class 2: x2 is large (above both "high" planes).
# Class 1: everything in between.
def net(x0, x1, x2, x3):
    low_a = x2 + 0.7 * x3 - 0.18 * x0 - 2 / 3 * x1
    low_b = x2 + 0.75 * x3 - x0 / 6 - 0.6 * x1
    if low_a < 0.1 and low_b < 0.46:
        return 0
    high_a = x2 + 1.8 * x3 - 0.36 * x0 - 0.62 * x1
    high_b = x2 + 1.5 * x3 - x0 / 3 - 0.8 * x1
    if high_a > 3.87 and high_b > 2.96:
        return 2
    return 1
