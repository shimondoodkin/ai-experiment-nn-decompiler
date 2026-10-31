# A round bump centred at the origin: about exp(-(x0*x0 + x1*x1)) (peak ~0.98 at (0,0), ~0.37 at radius 1).
# That formula alone misses the network by up to 0.06 (tolerance 0.02) because the network's own
# ripples (+-3..5%) are spread over the whole square, so the 16 tanh units are kept, snapped to
# 2 decimals, with the output layer refitted by least squares and rounded to 3 decimals.
def net(x0, x1):
    t0 = tanh(0.94 * x0 - 0.75 * x1 + 1.88)
    t1 = tanh(0.11 * x0 + 1.14 * x1 - 2.08)
    t2 = tanh(-1.53 * x0 + 0.43 * x1 - 0.83)
    t3 = tanh(-0.58 * x0 + 1.34 * x1 - 0.92)
    t4 = tanh(0.85 * x0 - 0.29 * x1 + 1.28)
    t5 = tanh(0.64 * x0 - 0.41 * x1 - 1.19)
    t6 = tanh(0.11 * x0 - 0.57 * x1 + 0.64)
    t7 = tanh(-0.3 * x0 - 0.26 * x1 - 0.57)
    t8 = tanh(1.06 * x0 + 0.15 * x1 + 1.72)
    t9 = tanh(1.28 * x0 + 0.24 * x1 - 0.63)
    t10 = tanh(-0.76 * x0 - 1.21 * x1 + 0.77)
    t11 = tanh(-0.43 * x0 - 1.15 * x1 - 0.61)
    t12 = tanh(-0.26 * x0 + 0.28 * x1 - 0.63)
    t13 = tanh(1.24 * x0 + 0.83 * x1 + 0.9)
    t14 = tanh(-0.84 * x0 + 0.77 * x1 + 0.66)
    t15 = tanh(-0.02 * x0 + 1.51 * x1 + 0.82)
    return -0.316 * t0 + 0.277 * t1 - 0.328 * t2 - 0.304 * t3 + 0.407 * t4 + 0.353 * t5 + 0.224 * t6 + 0.244 * t7 - 0.265 * t8 - 0.395 * t9 + 0.291 * t10 - 0.216 * t11 - 0.211 * t12 + 0.314 * t13 + 0.511 * t14 + 0.241 * t15 - 0.032
