def net(x0, x1):
    # A bump: peak about 1 at the origin, falling to about 0 beyond distance ~2.
    # Linear tilt left over after rewriting the mostly-active units as z + relu(-z).
    tilt = 1.03 - 0.495 * x0 + 0.44 * x1
    # Ridge fan through the origin (concave peak); every weight is negative.
    up_a = relu(x1 - 0.16 * x0 + 0.02)
    up_b = relu(x1 + 0.21 * x0 + 0.02)
    up_c = relu(x1 - 0.9 * x0 + 0.1)
    left_a = relu(0.3 * x1 - x0)
    left_b = relu(0.08 - x0 - 0.03 * x1)
    left_c = relu(0.05 - x0 - 0.28 * x1)
    down_left = relu(-0.9 * x0 - x1)
    ridge = 0.39 * up_a + 0.41 * up_b + 0.1 * up_c + 0.24 * left_a + 0.315 * left_b + 0.29 * left_c + 0.035 * down_left
    # Skirt: kinks about 2 away from the origin that flatten the far sides back to ~0.
    ne = relu(x0 + x1 - 2.1)
    nw = relu(x1 - x0 - 2.1)
    sw = relu(-0.95 * x0 - x1 - 2)
    se_a = relu(x0 - 0.6 * x1 - 1.7)
    se_b = relu(0.75 * x0 - x1 - 1.8)
    skirt = 0.42 * ne + 0.4 * nw + 0.415 * sw + 0.23 * se_a + 0.305 * se_b
    return tilt - ridge + skirt
