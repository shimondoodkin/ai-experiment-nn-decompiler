def net(x0, x1, x2, x3):
    # inputs match the iris ranges: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width
    # class 0 (setosa-like): petals small relative to sepal width
    small_petals = 0.8 * x1 + 1.4 - x2 - 1.2 * x3 - 0.05 * x0
    if small_petals > 0:
        return 0
    # class 2 (virginica-like): petals large; two linear pieces from the net's ReLU kink
    big_petals = x2 + 1.75 * x3 - 0.4 * x0 - 0.7 * x1 - 10 / 3
    big_petals_kink = x2 + 2 * x3 - 0.4 * x0 - 0.65 * x1 - 3.6
    if big_petals > 0 and big_petals_kink > 0:
        return 2
    return 1
