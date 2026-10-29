# Two-feature, three-class classifier. The 8 tanh units only draw two nearly straight
# boundaries that cut the input box into three diagonal bands:
#   class 0: small x0 and small x1   (x0 + 2.8*x1 < 4.5)
#   class 2: large x0 and/or x1      (x0 + x1 >= 6.6)
#   class 1: the band in between
def net(x0, x1):
    low_score = x0 + 2.8 * x1
    high_score = x0 + x1
    if low_score < 4.5:
        return 0
    if high_score < 6.6:
        return 1
    return 2
