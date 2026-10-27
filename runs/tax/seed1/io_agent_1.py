def net(x0):
    # Two-rate linear charge: 0.185 per unit above 6, plus an extra 0.215 per unit above 78
    # (i.e. the marginal rate rises from 0.185 to 0.4 at x0 = 78).
    base = 0.185 * (x0 - 6)
    surcharge = 0.215 * relu(x0 - 78)
    return max(base + surcharge, -0.75)
