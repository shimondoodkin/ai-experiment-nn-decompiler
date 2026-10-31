# Overall shape: a smoothed distance from the origin, roughly sqrt(x0*x0 + x1*x1 + 0.12) - 0.06.
# That formula alone is off by up to about 0.083 (the net wiggles by about +-0.08 around it, and is
# about 0.37 at the origin), which is more than the snap tolerance (0.078). So the program keeps the
# 10 sigmoid units that carry the shape, with weights snapped to 0.1, and replaces the 6 near-linear
# low-weight units (4, 6, 8, 10, 13, 15 of the dump) with one linear term. Output weights were refit
# (minimax) and rounded to 0.01.
def net(x0, x1):
    s0 = sigmoid(0.5 * x0 - 1.9 * x1 - 2.8)
    s1 = sigmoid(0.1 * x0 + 1.3 * x1 - 3.5)
    s2 = sigmoid(-1.6 * x0 + 1.3 * x1 - 2)
    s3 = sigmoid(0.7 * x0 + 1.9 * x1 - 1.7)
    s5 = sigmoid(1.8 * x0 - 4.4)
    s7 = sigmoid(0.1 * x0 - 1.4 * x1 - 3.8)
    s9 = sigmoid(2.4 * x0 - 1.8)
    s11 = sigmoid(0.8 * x0 - 3.4 * x1 - 1.6)
    s12 = sigmoid(-1.8 * x0 - 1.3 * x1 - 2)
    s14 = sigmoid(-1.2 * x0 - 3.1)
    return -0.56 - 0.19 * x0 - 0.03 * x1 + 0.88 * s0 + 2.01 * s1 + 1.01 * s2 + 1.24 * s3 + 2.01 * s5 + 1.61 * s7 + 1.3 * s9 + 0.48 * s11 + 1.01 * s12 + 1.37 * s14
