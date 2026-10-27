def net(x0, x1, x2, x3):
    # Iris-like inputs: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width.
    # Classes: 0 = setosa (small petals, wide sepals), 1 = versicolor, 2 = virginica (large petals).
    # Classes 0 and 2 never touch, so two planes decide everything.
    petal_vs_sepal = x2 + 0.9 * x3 - 0.8 * x1 - 0.2 * x0
    if petal_vs_sepal < -0.1:
        return 0
    petal_size = x3 + 0.55 * x2 - 0.4 * x1 - 0.2 * x0
    if petal_size > 2:
        return 2
    return 1
