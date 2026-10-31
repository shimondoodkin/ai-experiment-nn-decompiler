def net(x0, x1):
    # A radial bump centred at the origin, roughly exp(-(x0*x0 + x1*x1)) (peak 1 at (0, 0),
    # 0.37 at radius 1, ~0 beyond radius 2), built as 14 sigmoid ridges. The closed form alone
    # misses the net by up to 0.027 (ripples in the flat tails), so the ridges are kept.
    # Ridge slopes/offsets snapped to 0.1, output weights refitted (minimax) and snapped to 0.01.
    r0 = sigmoid(-0.8 * x0 + 2.6 * x1 - 1.4)
    r1 = sigmoid(-2 * x0 - 2.1 * x1 - 1.6)
    r2 = sigmoid(0.8 * x0 - 1.4 * x1 - 2)
    r3 = sigmoid(-1.3 * x0 + 0.7 * x1 - 1.7)
    r4 = sigmoid(-2.5 * x0 - 1.3 * x1 + 1.5)
    r5 = sigmoid(-2.7 * x0 + 0.7 * x1 + 1.4)
    r6 = sigmoid(-0.2 * x0 - 1.5 * x1 + 1.6)
    r7 = sigmoid(0.2 * x0 + 2.8 * x1 + 1.4)
    r8 = sigmoid(-2.7 * x0 - 0.3 * x1 - 1.3)
    r9 = sigmoid(1.1 * x0 + 2.6 * x1 - 1.5)
    r10 = sigmoid(-2.3 * x0 + 1.6 * x1 - 1.5)
    r11 = sigmoid(-x0 - 1.2 * x1 - 1.9)
    r12 = sigmoid(-1.6 * x0 - 0.4 * x1 + 2.1)
    r13 = sigmoid(-1.7 * x0 + 2.3 * x1 + 1.6)
    return 0.09 - 0.69 * r0 - 0.58 * r1 + 0.47 * r2 + 0.55 * r3 + 0.63 * r4 + 0.68 * r5 - 0.53 * r6 + 0.72 * r7 - 0.72 * r8 - 0.7 * r9 - 0.67 * r10 + 0.49 * r11 - 0.42 * r12 + 0.61 * r13
