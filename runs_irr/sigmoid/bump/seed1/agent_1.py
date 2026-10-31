# A single bump centred at the origin, height ~1, close to exp(-(x0*x0 + x1*x1)).
# The net builds it as 1.5 minus a ring of steep sigmoid "walls": each wall
# sigmoid(~3 * (u . x) - ~1.5) switches on when the point moves past ~0.5 along the
# direction u (angle in the name, degrees). Inside all walls the output is ~1; far out
# it falls to ~0. The net's own tail ripple (up to 0.05 away from exp(-r^2)) is kept by
# five shallow units, which is why the plain Gaussian formula does not pass.
def net(x0, x1):
    # ten steep walls, roughly evenly spaced around the circle, at radius ~0.5
    wall_000 = sigmoid(2.5 * x0 - 1.5)
    wall_043 = sigmoid(2.4 * x0 + 2.25 * x1 - 1.75)
    wall_081 = sigmoid(0.5 * x0 + 3 * x1 - 1.5)
    wall_116 = sigmoid(-1.5 * x0 + 3 * x1 - 1.5)
    wall_155 = sigmoid(-2.75 * x0 + 1.3 * x1 - 1.7)
    wall_177 = sigmoid(-3 * x0 + 0.2 * x1 - 2)
    wall_204 = sigmoid(-2.75 * x0 - 1.2 * x1 - 1.75)
    wall_241 = sigmoid(-1.5 * x0 - 2.75 * x1 - 1.5)
    wall_278 = sigmoid(0.4 * x0 - 3 * x1 - 1.5)
    wall_315 = sigmoid(1.9 * x0 - 1.9 * x1 - 1.5)
    walls = sum(0.58 * wall_000, 0.426 * wall_043, 0.435 * wall_081, 0.471 * wall_116,
                0.631 * wall_155, 0.346 * wall_177, 0.596 * wall_204, 0.481 * wall_241,
                0.508 * wall_278, 0.663 * wall_315)
    # five shallow units that shape the tails (they add back where the walls overshoot)
    tail_058 = sigmoid(0.9 * x0 + 1.5 * x1 - 3.75)
    tail_169 = sigmoid(-1.5 * x0 + 0.25 * x1 - 1.1)
    tail_201 = sigmoid(-1.5 * x0 - 0.5 * x1 - 1.75)
    tail_294 = sigmoid(0.5 * x0 - 0.75 * x1 - 1.5)
    tail_318 = sigmoid(x0 - x1 - 2)
    tails = sum(0.423 * tail_058, 0.742 * tail_169, 0.316 * tail_201, 0.295 * tail_294,
                0.509 * tail_318)
    return 1.496 - walls + tails
