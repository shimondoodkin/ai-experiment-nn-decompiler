# Two-input, three-class classifier. The 8 sigmoid units only carve two
# almost straight cuts through the (x0, x1) plane:
#   class 0  below  x0 + 2.5*x1 = 4.3   (small x0 and small x1)
#   class 2  above  x0 + 1.1*x1 = 6.7   (large x0 and/or large x1)
#   class 1  in the band between them.
# The two lines never meet inside the input box, so the regions are ordered.
def net(x0, x1):
    low_score = x0 + 2.5 * x1
    high_score = x0 + 1.1 * x1
    if low_score < 4.3:
        return 0
    if high_score < 6.7:
        return 1
    return 2
