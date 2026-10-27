def net(x0, x1, x2, x3):
    # iris-like inputs: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width
    small_petal = 0.1 * x0 - 0.8 * x1 + x2 + 1.05 * x3 < 1.6
    large_petal = -0.4 * x0 - 0.7 * x1 + x2 + 1.8 * x3 > 3.45
    # versicolor/virginica overlap zone: petal length about 5.0-5.4, petal width about 1.4-1.6.
    # Here the sepal-weighted score is unreliable; petal width alone decides.
    overlap_zone = 4.95 <= x2 < 5.5 and 1.35 <= x3 < 1.7
    if small_petal:
        return 0
    if overlap_zone:
        return 2 if x3 < 1.55 else 1
    if large_petal:
        return 2
    return 1
