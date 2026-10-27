def net(x0, x1, x2, x3):
    # Iris-like classifier: two linear tests (the 8-neuron hidden layer collapses to these).
    small_petal = x0 + 4 * x1 - 5 * x2 - 4.5 * x3        # > 0 -> class 0
    if small_petal > 0:
        return 0
    narrow_petal = 4 * x0 + 8 * x1 - 11 * x2 - 20 * x3 + 40   # >= 0 -> class 1, else class 2
    if narrow_petal >= 0:
        return 1
    return 2
