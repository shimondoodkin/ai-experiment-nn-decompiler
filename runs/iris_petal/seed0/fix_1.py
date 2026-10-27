def net(x0, x1):
    # Iris-like classifier on (petal length x0, petal width x1); two straight boundaries.
    # setosa / versicolor boundary: unchanged from the network.
    if x0 + 2.0516378 * x1 <= 4.0960478:
        return 0
    # versicolor / virginica boundary: steeper in petal width than the network's
    # x0 + 1.1*x1 > 6.8, i.e. 0.4*petal length + petal width > 3.49.
    if x0 + 2.5 * x1 > 8.725:
        return 2
    return 1
