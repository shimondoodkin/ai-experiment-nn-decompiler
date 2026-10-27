def net(x0, x1):
    # x0 ~ length-like feature in [1, 6.9], x1 ~ width-like feature in [0.1, 2.5]
    # class 0: small (below the line x0 + 9/4 * x1 = 4.4)
    small = x0 + 2.25 * x1 < 4.4
    if small:
        return 0
    # class 2: large, by the combined size, by width alone, or by x0 alone
    large = x0 + 1.1 * x1 > 6.6 or x1 > 1.65 or x0 > 6.3
    if large:
        return 2
    return 1
