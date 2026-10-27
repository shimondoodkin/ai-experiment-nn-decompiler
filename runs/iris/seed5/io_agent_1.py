def net(x0, x1, x2, x3):
    # Iris-like inputs: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width.
    # Classes: 0 setosa, 1 versicolor (default), 2 virginica.
    sepal_len = x0
    sepal_wid = x1
    petal_len = x2
    petal_wid = x3
    # setosa vs versicolor: one hyperplane (≈ out0 - out1)
    setosa = 4 * sepal_wid - 5 * petal_len - 6 * petal_wid - 0.2 * sepal_len + 6.8
    # virginica vs versicolor: ≈ out2 - out1 is min of two planes (one ReLU kink);
    # the main plane, and a second plane that only bites in the
    # narrow-sepal / long-petal / thin-petal corner.
    virg_main = -3.3 * sepal_len - 6 * sepal_wid + 8.5 * petal_len + 15 * petal_wid - 29
    virg_corner = -2.5 * sepal_len - 4 * sepal_wid + 7 * petal_len + 13 * petal_wid - 27.5
    if setosa > 0:
        return 0
    elif virg_main > 0 and virg_corner > 0:
        return 2
    return 1
