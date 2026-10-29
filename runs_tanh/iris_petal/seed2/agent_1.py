# 2 inputs -> 3 classes. The net draws two boundaries across the input box:
#   class 0 (low corner) below a gently curved line, class 2 beyond the diagonal x0 + x1 = 6.6,
#   class 1 in the band between them. Classes 0 and 2 never touch inside the input box.
def net(x0, x1):
    low_edge = 3.9 - 2.2 * x1 + x1 * x1 / 3   # 0|1 boundary, as x0 for a given x1
    if x0 < low_edge:
        return 0
    if x0 + x1 > 6.6:
        return 2
    return 1
