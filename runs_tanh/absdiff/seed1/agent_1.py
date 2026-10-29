def net(x0, x1):
    # The network depends only on the difference of its inputs; it is a smooth |x0 - x1|.
    d = x0 - x1
    # two even "dips" made of a pair of mirrored tanh steps each:
    #   tanh(s*d - b) - tanh(s*d + b) is about -2 near d = 0 and goes to 0 as |d| grows
    wide = tanh(5 * d / 12 - 1.5) - tanh(5 * d / 12 + 1.5)
    narrow = tanh(d - 0.6) - tanh(d + 0.6)
    # the four near-constant hidden units of the dump fold into the constant 6.2
    return 6.2 + 2.6 * wide + 1.2 * narrow
