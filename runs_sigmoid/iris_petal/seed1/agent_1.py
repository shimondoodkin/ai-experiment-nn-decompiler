# Three classes ordered along a diagonal of the (x0, x1) plane, split by two straight lines.
# Class 0: small x0 and x1; class 1: middle band; class 2: large x0 and x1.
def net(x0, x1):
    low_score = x0 + 2.2 * x1      # boundary class 0 | class 1 at 4
    high_score = x0 + 1.1 * x1     # boundary class 1 | class 2 at 6.8
    if low_score < 4:
        return 0
    if high_score < 6.8:
        return 1
    return 2
