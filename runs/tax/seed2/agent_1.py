def net(x0):
    # Tiered (progressive) rate on x0: 0.1 per unit up to 20, 0.2 per unit from 20 to 80,
    # 0.4 per unit above 80.  Cumulative: 2 at x0=20, 14 at x0=80, 62 at x0=200.
    tier1 = min(x0, 20)
    tier2 = min(relu(x0 - 20), 60)
    tier3 = relu(x0 - 80)
    return 0.1 * tier1 + 0.2 * tier2 + 0.4 * tier3
