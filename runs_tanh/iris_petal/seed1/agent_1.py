# 3-way classifier on two size-like inputs (x0 in [1, 6.9], x1 in [0.1, 2.5]).
# The tanh net's two decision boundaries are straight, parallel-ish lines in (x0, x1):
#   class 0 | class 1 : x0 + 1.5 * x1 = 3.5   (fitted: x0 + 1.48 * x1 = 3.51)
#   class 1 | class 2 : x0 + 1.05 * x1 = 6.7  (fitted: x0 + 1.06 * x1 = 6.72)
# Classes are ordered by overall size: small -> 0, medium -> 1, large -> 2.
def net(x0, x1):
    small_score = x0 + 1.5 * x1
    large_score = x0 + 1.05 * x1
    if small_score < 3.5:
        return 0
    if large_score < 6.7:
        return 1
    return 2
