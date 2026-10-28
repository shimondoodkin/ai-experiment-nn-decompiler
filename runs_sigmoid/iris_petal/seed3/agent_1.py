# 3-class classifier over two inputs, decided by two straight cuts in the (x0, x1) plane.
# Class 0 = "both small" corner, class 2 = "both large" corner, class 1 = the band between.
def net(x0, x1):
    low_score = x0 + 4.25 * x1      # steep cut: x1 matters ~4x more than x0
    high_score = x0 + 1.1 * x1      # shallow cut: x0 and x1 weigh about the same
    if low_score < 5.5:
        return 0
    if high_score > 6.75:
        return 2
    return 1
