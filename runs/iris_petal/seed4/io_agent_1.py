def net(x0, x1):
    # Iris-like classifier: x0 = petal length, x1 = petal width (cm).
    # 0 = setosa, 1 = versicolor, 2 = virginica.
    petal_length = x0
    petal_width = x1
    # setosa / versicolor boundary: length + width = 3.25
    if petal_length + petal_width < 3.25:
        return 0
    # versicolor / virginica boundary: length + (13/12) width = 6.75,
    # i.e. 12*length + 13*width = 81
    if 12 * petal_length + 13 * petal_width < 81:
        return 1
    return 2
