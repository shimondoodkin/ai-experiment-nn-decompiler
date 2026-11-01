# A soft round bump centred at the origin: close to exp(-(x0*x0 + x1*x1)) (peak ~1.01 at 0,
# ~0.37 at radius 1, ~0 beyond radius 2). The clean formula misses the net by up to 0.030 in the
# corners of the box (the net's ripple there), beyond the 0.021 snap tolerance, so the program
# keeps the net's ridge structure with weights rounded to 2 decimals.
#
# Each line is "inside a half-plane" signal tanh(k * (u . x) - c), sign-normalised so c > 0:
#  s0..s10: 11 sharp walls (k ~ 1.4, c ~ 0.63, i.e. at distance ~0.45 from the origin), spread
#           around the circle about every 33 degrees; weight ~ -0.3, so the bump is a soft
#           11-gon of half-plane indicators.
#  w0..w4:  5 soft walls (k ~ 0.7, c ~ 0.4..1.16), about every 72 degrees, weight ~ +0.27,
#           rounding the polygon's shoulders.
def net(x0, x1):
    s0 = tanh(1.30 * x0 + 0.52 * x1 - 0.61)
    s1 = tanh(0.77 * x0 + 0.99 * x1 - 0.58)
    s2 = tanh(0.12 * x0 + 1.39 * x1 - 0.65)
    s3 = tanh(-0.68 * x0 + 1.17 * x1 - 0.62)
    s4 = tanh(-1.21 * x0 + 0.62 * x1 - 0.61)
    s5 = tanh(-1.31 * x0 - 0.15 * x1 - 0.64)
    s6 = tanh(-1.11 * x0 - 0.84 * x1 - 0.58)
    s7 = tanh(-0.51 * x0 - 1.22 * x1 - 0.65)
    s8 = tanh(0.31 * x0 - 1.41 * x1 - 0.68)
    s9 = tanh(0.99 * x0 - 0.95 * x1 - 0.66)
    s10 = tanh(1.41 * x0 - 0.26 * x1 - 0.65)
    w0 = tanh(0.52 * x0 + 0.49 * x1 - 0.91)
    w1 = tanh(-0.20 * x0 + 0.55 * x1 - 0.40)
    w2 = tanh(-0.69 * x0 - 0.02 * x1 - 0.82)
    w3 = tanh(-0.21 * x0 - 0.71 * x1 - 0.82)
    w4 = tanh(0.59 * x0 - 0.53 * x1 - 1.16)
    return 0.03 - 0.26 * s0 - 0.32 * s1 - 0.31 * s2 - 0.34 * s3 - 0.31 * s4 - 0.31 * s5 - 0.28 * s6 - 0.33 * s7 - 0.28 * s8 - 0.27 * s9 - 0.28 * s10 + 0.26 * w0 + 0.34 * w1 + 0.28 * w2 + 0.26 * w3 + 0.22 * w4
