def net(x0, x1, x2, x3):
    # Iris-like flower classifier: x0 sepal length, x1 sepal width,
    # x2 petal length, x3 petal width.  0 = setosa, 1 = versicolor, 2 = virginica.
    # Each boundary is "petal size minus a bit of sepal size" against a threshold.
    petal_vs_sepal_small = x3 + 0.9*x2 - 0.55*x1 - 0.2*x0 - 0.5
    petal_vs_sepal_large = x3 + 0.55*x2 - 0.4*x1 - 0.2*x0 - 2.05
    if petal_vs_sepal_small <= 0:
        return 0
    if petal_vs_sepal_large <= 0:
        return 1
    return 2
