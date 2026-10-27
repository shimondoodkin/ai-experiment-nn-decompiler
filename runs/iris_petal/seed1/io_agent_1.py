def net(x0, x1):
    # x0 ~ petal length, x1 ~ petal width (Iris-like); classes 0/1/2 ~ setosa/versicolor/virginica
    size = x0 + x1
    if size > 6.65:
        return 2
    if x0 + 1.25 * x1 < 3.45:
        return 0
    return 1
