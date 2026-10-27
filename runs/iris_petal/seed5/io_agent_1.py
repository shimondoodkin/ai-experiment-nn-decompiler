def net(x0, x1):
    # Iris-like classifier on petal length (x0, cm) and petal width (x1, cm).
    petal_length = x0
    petal_width = x1
    # class 0 (setosa): small petals; oblique cut petal_length + 1.75*petal_width < 3.85
    if petal_length + 1.75 * petal_width < 3.85:
        return 0
    # class 2 (virginica): large petals; petal_length + petal_width >= 6.65
    if petal_length + petal_width < 6.65:
        return 1
    return 2
