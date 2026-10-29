def net(x0, x1, x2, x3):
    # six tanh features; the score of class 1 is fixed at 0, classes 0 and 2 are scored against it
    small = tanh(0.325 * x0 + 1.3 * x1 - 1.8 * x2 - 2.4 * x3 + 1.2)       # high when x2, x3 are small
    big_a = tanh(0.1 * x0 - 1.2 * x1 + 0.7 * x2 + 0.9 * x3 - 1.6)
    big_b = tanh(-0.2 * x0 - 0.7 * x1 + 0.7 * x2 + 1.4 * x3 - 2.6)       # high when x3 is large
    big_c = tanh(-0.6 * x0 - 0.45 * x1 + 1.1 * x2 + 1.4 * x3 - 2.1)
    long_x2 = tanh(-0.1 * x0 - 0.9 * x1 + 1.4 * x2 + 0.6 * x3 - 1)      # mostly x2 against x1
    narrow_x1 = tanh(0.15 * x0 - 1.3 * x1 + 0.2 * x2 - 0.4 * x3 + 1.9)   # high when x1 is small
    score0 = 5.5 * small + 0.75 * big_b + 1.25 * big_c - 4 * long_x2 - 2.25 * narrow_x1 - 0.25
    score2 = 2 * small + 3.5 * big_a + 6.5 * big_b + 4.75 * big_c - 2 * long_x2 - 2.75 * narrow_x1 - 1
    return argmax(score0, 0, score2)
