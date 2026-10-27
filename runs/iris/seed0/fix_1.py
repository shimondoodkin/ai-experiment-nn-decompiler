def net(x0, x1, x2, x3):
    # Iris-like classifier: two linear tests, plus one override for the versicolor/virginica overlap.
    small_petal = x0 + 4 * x1 - 5 * x2 - 4.5 * x3        # > 0 -> class 0
    if small_petal > 0:
        return 0
    # Overlap zone: petal length just past 5 with width below 1.75. The linear test below is
    # unreliable here; petal width alone decides (1.6-1.7 -> class 1, up to 1.5 -> class 2).
    if 4.95 < x2 < 5.25 and x3 < 1.75:
        return 1 if x3 > 1.55 else 2
    narrow_petal = 4 * x0 + 8 * x1 - 11 * x2 - 20 * x3 + 40   # >= 0 -> class 1, else class 2
    if narrow_petal >= 0:
        return 1
    return 2
