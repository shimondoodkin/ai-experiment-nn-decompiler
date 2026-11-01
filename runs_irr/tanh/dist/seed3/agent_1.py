def net(x0, x1):
    # The net approximates the distance from the origin, sqrt(x0*x0 + x1*x1),
    # smoothed near (0, 0) (the net gives about 0.46 there) and with small wiggles.
    # Seven tanh ridges remain; the nine near-flat units of the dump became the
    # constant and the -0.03*x1 slope.
    # ridges facing -x1 (rise of the distance for x1 < 0)
    down = tanh(-0.2*x0 - 0.7*x1 - 0.6)
    down_far = tanh(-0.1*x0 - 0.4*x1 - 1.3)
    # ridges facing +x1
    up = tanh(-0.1*x0 + 0.44*x1 - 0.6)
    # diagonal ridges
    diag_pp = tanh(0.85*x0 + 0.7*x1 - 1)
    diag_mp = tanh(-0.7*x0 + 0.65*x1 + 0.95)
    # a pair of steps in x0 forming a smooth |x0|-like valley
    left = tanh(0.5*x0 + 0.7)
    right = tanh(0.5*x0 - 0.9)
    return 6.41 - 0.03*x1 + down + 1.5*down_far + 2.3*up + 0.45*diag_pp - 0.6*diag_mp - 1.51*left + 1.65*right
