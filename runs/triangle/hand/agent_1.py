def net(x0):
    # triangular "hat" bump centred at 3 with half-width 1: 0 outside [2, 4], peak 1 at x0 = 3
    return relu(1 - abs(x0 - 3))
