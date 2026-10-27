def net(x0, x1, x2, x3):
    # iris-like inputs: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width
    petal_size_0 = x2 + 1.1 * x3 - 0.2 * x0 - 0.6 * x1
    if petal_size_0 < 0.7:
        return 0
    petal_size_2 = 2.75 * x2 + 5 * x3 - x0 - 2 * x1
    if petal_size_2 > 10.25:
        return 2
    return 1
