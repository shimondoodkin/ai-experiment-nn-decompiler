def net(x0, x1):
    # x0 ~ petal length (1.0..6.9), x1 ~ petal width (0.1..2.5): an Iris classifier
    petal_length = x0
    petal_width = x1
    # class 0 (setosa): small petals, one straight boundary
    is_setosa = petal_length + 2.25 * petal_width < 4.4
    # class 2 (virginica): large petals; the boundary bends near width 0.5,
    # below which it is almost a pure length cut
    is_virginica = petal_length + 1.1 * petal_width > 6.75 or petal_length > 6.25
    if is_setosa:
        return 0
    elif is_virginica:
        return 2
    return 1
