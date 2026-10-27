def net(x0, x1, x2, x3):
    # iris-like inputs: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width
    small_petal = 0.1 * x0 - 0.8 * x1 + x2 + 1.05 * x3 < 1.6
    large_petal = -0.4 * x0 - 0.7 * x1 + x2 + 1.8 * x3 > 3.45
    if small_petal:
        return 0
    if large_petal:
        return 2
    return 1
