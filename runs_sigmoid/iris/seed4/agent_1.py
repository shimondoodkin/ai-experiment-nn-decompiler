# Three classes separated by two straight cuts (the net's 8 sigmoid units fold into these).
# Both cuts weigh the inputs the same way: x1 and x0 push up, x2 and x3 push down.
def net(x0, x1, x2, x3):
    # class 0: small x2 + x3 compared with x1 (and a little x0)
    low_cut = x0 + 6 * x1 + 3.1 - 8 * (x2 + x3)
    # class 2: large x2 and x3 (x3 counts 1.65x) compared with x0 and x1
    high_cut = 20 * x2 + 33 * x3 - 8 * x0 - 14 * x1 - 63
    if low_cut > 0:
        return 0
    if high_cut > 0:
        return 2
    return 1
