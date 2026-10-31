def net(x0, x1):
    # A single bump centred at the origin: about 0.95 at (0, 0), about 0.37 at distance 1
    # (roughly exp(-(x0^2 + x1^2))), flattening to about 0 (small +-0.1 ripples) beyond distance 2.
    # Built as a sum of ramps (one ReLU per crease line).
    # Creases through (or next to) the origin: they carve the peak.
    peak_e = relu(x0 + 0.1)
    peak_ese = relu(x0 - 0.55 * x1 + 0.1)
    peak_nw = relu(-0.45 * x0 + x1 + 0.05)
    peak_ne = relu(0.55 * x0 + x1)
    peak_se = relu(0.65 * x0 - x1 + 0.1)
    peak_wsw = relu(-x0 - 0.45 * x1 - 0.1)
    peak_s = relu(-0.3 * x0 - x1)
    peak_sw = relu(-x0 - 0.8 * x1 - 0.35)
    # Creases at distance about 1.25: they bend the slopes back to flat tails.
    rim_low_x1 = relu(0.02 * x0 - x1 - 1.25)
    rim_high_x1 = relu(0.02 * x0 + x1 - 1.2748)
    rim_high_x0 = relu(-x0 + 1.25)
    rim_low_x0 = relu(x0 - 0.05 * x1 + 1.25)
    rim_low_x0_a = relu(x0 - 0.5 * x1 + 1.25)
    rim_low_x0_b = relu(x0 + 0.45 * x1 + 1.3)
    peak = sum(-0.23 * peak_e, -0.316 * peak_ese, -0.298 * peak_nw, -0.332 * peak_ne, -0.155 * peak_se, -0.236 * peak_wsw, -0.11 * peak_s, -0.118 * peak_sw)
    rim = sum(0.468 * rim_low_x1, 0.463 * rim_high_x1, 0.422 * rim_high_x0, 0.246 * rim_low_x0, 0.18 * rim_low_x0_a, 0.228 * rim_low_x0_b)
    return peak + rim - 0.319
