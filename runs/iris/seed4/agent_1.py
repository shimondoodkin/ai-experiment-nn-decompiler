# Iris-like 3-class classifier (x0 sepal length, x1 sepal width, x2 petal length, x3 petal width).
# Of the 8 hidden ReLUs only 4 matter and they are active almost everywhere, so the net is
# effectively linear: two independent half-space tests decide the class.
def net(x0, x1, x2, x3):
    small_petals = x1 + 4.5 - 2 * x2 - 3 * x3          # class 0 ("setosa") score
    if small_petals > 0:
        return 0
    large_petals = 8 * x2 + 14 * x3 - 3 * x0 - 6 * x1 - 27   # class 2 ("virginica") score
    if large_petals > 0:
        return 2
    return 1
