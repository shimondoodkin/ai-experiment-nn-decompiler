# Binary classifier: class 1 when the two inputs are within 0.2 of each other.
def net(x0, x1):
    gap = abs(x1 - x0)
    return 0.2 - gap
