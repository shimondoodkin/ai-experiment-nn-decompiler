def net(x0, x1):
    # relu(d) + relu(-d) == |d|, with d = x0 - x1
    return abs(x0 - x1)
