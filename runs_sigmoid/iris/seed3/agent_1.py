def net(x0, x1, x2, x3):
    # The net's three argmax regions are cut by two planes (each found by a max-margin fit
    # on the net's confident decisions; no rounder coefficients keep a positive margin).
    # Class 0: x2 and x3 small compared with x1.
    first_class_score = x2 + 0.94 * x3 - 0.88 * x1 - 0.02 * x0
    if first_class_score <= 0.5:
        return 0
    # Class 2 vs class 1: x3 (and x2) large compared with x1 and x0.
    # (this plane has the thinnest margin, about 0.03; 2-decimal coefficients flip a corner point)
    last_class_score = x3 + 0.597 * x2 - 0.412 * x1 - 0.236 * x0
    if last_class_score > 1.953:
        return 2
    return 1
