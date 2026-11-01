def net(x0):
    # exponential decay: starts at 3 for x0 = 0 and halves every 2*ln(2) ~ 1.386 units
    return 3 * exp(-x0 / 2)
