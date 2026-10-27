def net(x0, x1, x2, x3):
    # Iris-like inputs: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width.
    # Classes: 0 = setosa, 1 = versicolor, 2 = virginica.
    sepal_length = x0
    sepal_width = x1
    petal_length = x2
    petal_width = x3
    # Positive on the versicolor side, negative on the virginica side.
    versicolor_score = 3 * sepal_length + 6 * sepal_width - 8.5 * petal_length - 15.5 * petal_width + 30.75
    # Positive for small petals with wide sepals.
    setosa_score = -sepal_length + 4.2 * sepal_width - 4.3 * petal_length - 6.7 * petal_width + 9.3
    if versicolor_score < 0:
        return 2
    if setosa_score > 0:
        return 0
    return 1
