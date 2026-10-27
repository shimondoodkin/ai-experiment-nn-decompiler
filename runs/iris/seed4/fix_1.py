# Iris-like 3-class classifier (x0 sepal length, x1 sepal width, x2 petal length, x3 petal width).
# Repaired version of the decompiled net: the same two linear tests, plus one correction close to the
# class-1/class-2 boundary for long petals, where petal width decides instead of the linear score.
def net(x0, x1, x2, x3):
    small_petals = x1 + 4.5 - 2 * x2 - 3 * x3          # class 0 ("setosa") score
    if small_petals > 0:
        return 0
    large_petals = 8 * x2 + 14 * x3 - 3 * x0 - 6 * x1 - 27   # class 2 ("virginica") score
    if x2 > 4.95 and x3 < 1.75 and abs(large_petals) < 3:    # long, not-wide petals, borderline score
        return 2 if x3 < 1.55 else 1                         # narrow -> virginica, medium -> versicolor
    if large_petals > 0:
        return 2
    return 1
