def net(x0):
    # Convex piecewise-linear curve with 3 segments (looks like a coarse exp(x) fit).
    # Segment slopes: 0.43908 (x < 0.09704), 2.01913 (0.09704..1.22568), 5.10802 (x > 1.22568).
    base = 0.8623605 + 0.4390840 * x0
    knee1 = 1.5800445 * relu(x0 - 0.0970374)
    knee2 = 3.0888949 * relu(x0 - 1.2256840)
    return base + knee1 + knee2
