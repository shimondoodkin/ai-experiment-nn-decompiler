def net(x0, x1, x2, x3):
    # Iris-like classifier: 0 = setosa, 1 = versicolor, 2 = virginica
    sepal_len = x0
    sepal_wid = x1
    petal_len = x2
    petal_wid = x3
    # small petals (relative to sepal width) -> setosa
    petal_size = petal_len + 1.1 * petal_wid + 0.1 * sepal_len
    if petal_size < 0.8 * sepal_wid + 1.65:
        return 0
    # large petals (relative to sepal size) -> virginica
    big_petal = petal_len + 1.8 * petal_wid
    if big_petal > 0.4 * sepal_len + 0.7 * sepal_wid + 3.4:
        return 2
    return 1
