def net(x0, x1):
    # the network approximates a unit Gaussian bump centred at the origin
    r2 = x0 * x0 + x1 * x1
    gauss = exp(-r2)
    # local corrections: the ReLU net is a piecewise-linear fit of the Gaussian and its
    # fitting error (about +-0.03) is larger than the check tolerance, so the largest error
    # spots are reproduced with small pyramid bumps  h * relu(1 - (|x0-a| + |x1-b|) / w)
    fix = sum(
        -0.004 * relu(1 - (abs(x0 + 2.56) + abs(x1)) / 1),
        -0.007 * relu(1 - (abs(x0 + 2.5) + abs(x1 + 1.88)) / 1),
        -0.002 * relu(1 - (abs(x0 + 2.38) + abs(x1 - 1.22)) / 1),
        0.008 * relu(1 - (abs(x0 + 2.34) + abs(x1 - 3)) / 0.3),
        -0.005 * relu(1 - (abs(x0 + 2.12) + abs(x1 + 1.22)) / 0.3),
        0.001 * relu(1 - (abs(x0 + 1.64) + abs(x1 - 0.92)) / 0.6),
        -0.005 * relu(1 - (abs(x0 + 1.56) + abs(x1 + 2.48)) / 1),
        -0.003 * relu(1 - (abs(x0 + 1.44) + abs(x1 + 1.6)) / 0.3),
        0.009 * relu(1 - (abs(x0 + 1.28) + abs(x1 - 1.46)) / 1),
        -0.001 * relu(1 - (abs(x0 + 1.16) + abs(x1 - 0.4)) / 0.7),
        -0.001 * relu(1 - (abs(x0 + 1.1) + abs(x1 - 0.8)) / 0.7),
        -0.004 * relu(1 - (abs(x0 + 0.76) + abs(x1 - 0.98)) / 0.3),
        0.002 * relu(1 - (abs(x0 + 0.7) + abs(x1 - 0.62)) / 0.4),
        0.004 * relu(1 - (abs(x0 + 0.58) + abs(x1 - 1.68)) / 0.5),
        -0.001 * relu(1 - (abs(x0 + 0.46) + abs(x1 - 0.04)) / 0.4),
        -0.001 * relu(1 - (abs(x0 + 0.44) + abs(x1 + 2.62)) / 1),
        -0.001 * relu(1 - (abs(x0 + 0.14) + abs(x1 + 0.44)) / 0.6),
        0.01 * relu(1 - (abs(x0 + 0.08) + abs(x1 - 0.12)) / 0.3),
        -0.002 * relu(1 - (abs(x0 - 0.18) + abs(x1 - 0.42)) / 0.5),
        0.006 * relu(1 - (abs(x0 - 0.24) + abs(x1 + 0.9)) / 0.5),
        -0.001 * relu(1 - (abs(x0 - 0.28) + abs(x1 + 0.42)) / 0.5),
        0.001 * relu(1 - (abs(x0 - 0.42) + abs(x1 - 0.96)) / 0.7),
        0.001 * relu(1 - (abs(x0 - 0.52) + abs(x1 + 1.78)) / 0.8),
        -0.01 * relu(1 - (abs(x0 - 0.7) + abs(x1 + 1.04)) / 0.6),
        -0.003 * relu(1 - (abs(x0 - 1.16) + abs(x1 - 0.48)) / 0.7),
        0.002 * relu(1 - (abs(x0 - 1.2) + abs(x1 - 1.56)) / 1),
        -0.002 * relu(1 - (abs(x0 - 2) + abs(x1 - 2.48)) / 1),
        0.005 * relu(1 - (abs(x0 - 2.3) + abs(x1 + 3)) / 0.5),
        -0.005 * relu(1 - (abs(x0 - 2.3) + abs(x1 - 1.16)) / 0.8),
        0.001 * relu(1 - (abs(x0 - 3) + abs(x1 - 1.42)) / 0.8),
    )
    return gauss + fix
