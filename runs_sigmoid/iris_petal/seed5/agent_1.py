# 3 classes split by two nearly parallel straight lines in the (x0, x1) plane:
# class 0 below the low line, class 2 above the high line, class 1 in the band between.
def net(x0, x1):
    low_score = x0 + 2.7 * x1      # class 0 / class 1 line
    high_score = x0 + 1.125 * x1   # class 1 / class 2 line (1.125 = 9/8)
    if low_score < 4.45:
        return 0
    if high_score > 6.8:
        return 2
    return 1
