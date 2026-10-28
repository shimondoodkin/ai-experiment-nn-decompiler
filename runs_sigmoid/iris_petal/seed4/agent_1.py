# Two parallel-ish linear cuts split the (x0, x1) plane into three bands.
# Class 0: low corner (small x0 and x1); class 2: high corner; class 1 in between.
def net(x0, x1):
    low_score = 5 * x0 + 8 * x1      # boundary 0|1: x0 + 1.6*x1 = 3.6
    high_score = 8 * x0 + 9 * x1     # boundary 1|2: x0 + 1.125*x1 = 6.75
    if low_score < 18:
        return 0
    if high_score > 54:
        return 2
    return 1
