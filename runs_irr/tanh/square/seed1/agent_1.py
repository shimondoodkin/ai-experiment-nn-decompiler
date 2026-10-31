# A smooth "U" (bowl) centred at x0 = 0, close to x0*x0 on [-2, 2]:
# the difference of two tanh steps placed at x0 = +1.8 and x0 = -1.8, slope 5/6.
# The six near-linear tanh units of the dump contribute only a constant (~6.8)
# plus a tiny slope (~0.1); together with the output bias they fold into 7.25.
def net(x0):
    rise_right = tanh(5 / 6 * x0 - 1.5)   # step up around x0 = +1.8
    rise_left = tanh(5 / 6 * x0 + 1.5)    # step up around x0 = -1.8
    return 4 * (rise_right - rise_left) + 7.25
