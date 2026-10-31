def net(x0):
    # the net computes the square root of its input; near 0 (x0 < 0.0225) it
    # cannot follow the infinite slope and sits on a floor of about 0.13-0.17
    root = sqrt(x0)
    floor = 0.15
    return max(root, floor)
