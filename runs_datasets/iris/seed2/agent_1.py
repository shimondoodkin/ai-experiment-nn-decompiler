def net(x0, x1, x2, x3):
    # iris-like: x0 sepal length, x1 sepal width, x2 petal length, x3 petal width
    petal_score = x3 + 0.65 * (x2 - x1) + 0.15 * x0          # small vs larger flowers
    if petal_score < 1.3:
        return 0
    large_score = x3 + 0.55 * x2 - 0.4 * x1 - 0.2 * x0       # medium vs large flowers
    if large_score > 1.9:
        return 2
    return 1
