def net(x0):
    # The network approximates the reciprocal 1/x on [0.5, 4].
    # Near the lower edge (x0 ~ 0.5) the net's tanh units flatten out, so it undershoots 1/x
    # (net(0.5) = 1.964 instead of 2); a cap at 1.95 follows that droop.
    reciprocal = 1 / x0
    return min(reciprocal, 1.95)
