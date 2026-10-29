def net(x0):
    # The network's score stays in [-1.147, -1.100] over the whole input range [-2, 2]
    # (slope about -0.012 per unit of x0), always below the threshold 0.
    # So the decision never changes: the net always outputs class 0.
    return -1
