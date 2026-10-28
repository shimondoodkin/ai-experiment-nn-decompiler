# 3-class classifier over 4 measurements.
# All 8 sigmoid units point roughly along one direction (+x0, +x1, -x2, -x3); outside the
# net's unsure zone the three classes are cut apart by two straight boundaries:
#   class 0 | class 1   and   class 1 | class 2   (classes 0 and 2 never touch).
def net(x0, x1, x2, x3):
    # class 0: the x2 + x3 sum is small compared with x0 and x1
    if x2 + x3 < 0.23 * x0 + 0.58 * x1 + 0.367:
        return 0
    # class 2: x2 + 1.62 * x3 is large compared with x0 and x1
    if x2 + 1.62 * x3 > 0.4 * x0 + 0.7 * x1 + 3.217:
        return 2
    return 1
