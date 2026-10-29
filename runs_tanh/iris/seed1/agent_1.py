# Decision list over two piecewise-linear boundaries (each boundary = AND of two half-planes).
# Inputs look like the Iris measurements: x0 sepal length, x1 sepal width,
# x2 petal length, x3 petal width (cm). Classes: 0 small petals, 1 middle, 2 large petals.
def net(x0, x1, x2, x3):
    # class 2: large petal size relative to sepal size; two half-planes (the net's boundary is curved)
    big_a = 12 * x2 + 24 * x3 - 3 * x0 - 11 * x1
    big_b = 11 * x2 + 19 * x3 - 5 * x0 - 6 * x1
    if big_a > 50.4 and big_b > 38.2:
        return 2
    # class 0: small petals relative to sepals; two half-planes
    small_a = 6 * (x2 + x3) - 2 * x0 - 3 * x1
    small_b = 17 * x2 + 19 * x3 - 4 * x0 - 11 * x1
    if small_a < 0.7 and small_b < 3.36:
        return 0
    return 1
