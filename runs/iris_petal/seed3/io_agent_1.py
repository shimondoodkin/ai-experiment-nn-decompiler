def net(x0, x1):
    # Iris-like classifier on petal length (x0) and petal width (x1).
    # Classes: 0 = setosa (small petals), 1 = versicolor, 2 = virginica (large petals).
    # Each boundary is a straight line that bends once (max of two nearly parallel lines).
    # Snapped reading (PASS snap):  setosa if length + 3.2*width < 4.95,
    #                               virginica if length + 1.1*width > 6.8 or length + 0.7*width > 6.6
    petal_length = x0
    petal_width = x1
    size_vs_setosa = max(petal_length + 3.1219 * petal_width - 4.8907,
                         petal_length + 3.2452 * petal_width - 4.9741)
    if size_vs_setosa < 0:
        return 0
    size_vs_virginica = max(petal_length + 1.0913 * petal_width - 6.7805,
                            petal_length + 0.7018 * petal_width - 6.5783)
    if size_vs_virginica > 0:
        return 2
    return 1
