def net(x0, x1, x2, x3):
    # Iris-like classifier: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width
    # classes: 0 setosa, 1 versicolor, 2 virginica
    setosa_score = 4.5 + x1 - 2 * x2 - 3 * x3
    virginica_score = x3 + 0.56 * x2 - 0.4 * x1 - 0.22 * x0 - 1.94
    if setosa_score > 0:
        return 0
    if virginica_score > 0:
        return 2
    return 1
