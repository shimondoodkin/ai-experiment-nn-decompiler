def net(x0, x1):
    # A single bump centred on the origin: about 1.05 at (0, 0), falling roughly linearly to
    # about 0 at radius 1.4-1.6, and nearly flat (|y| < 0.1) further out. It looks like a
    # piecewise-linear approximation of exp(-(x0*x0 + x1*x1)); that formula alone is 0.115 off.
    # The ReLU pieces are kept because the net's own polygonal facets are what must be matched.

    # slopes going down from the peak: kinks through (or very near) the origin
    s_se = relu(0.5 * x0 - 0.5 * x1)
    s_sse = relu(0.4 * x0 - 0.6 * x1 + 0.05)
    s_e = relu(1 * x0 + 0.06 * x1)
    s_ne = relu(0.45 * x0 + 0.95 * x1)
    s_nw = relu(-0.35 * x0 + 0.8 * x1)
    s_w = relu(-1 * x0 + 0.25 * x1 - 0.15)
    s_sw = relu(-0.6 * x0 - 0.3 * x1 - 0.12)
    down = 0.2 * s_se + 0.25 * s_sse + 0.38 * s_e + 0.63 * s_ne + 0.5 * s_nw + 0.25 * s_w + 0.4 * s_sw

    # foot of the bump: kinks about 1-1.3 from the origin that cancel the slopes outside
    f_w = relu(0.5 * x0 + 0.05 * x1 + 0.6)
    f_n = relu(0.02 * x0 - 0.92 * x1 + 1.12)
    f_e = relu(-0.74 * x0 + 0.4 * x1 + 1.1)
    f_sw = relu(0.6 * x0 + 0.5 * x1 + 0.87)
    f_s = relu(-0.02 * x0 + 0.8 * x1 + 0.95)
    f_ne = relu(0.65 * x0 + 0.45 * x1 - 1)
    f_nw = relu(-0.73 * x0 + 0.49 * x1 - 1.03)
    foot = 0.45 * f_w + 0.447 * f_n + 0.5 * f_e + 0.44 * f_sw + 0.5 * f_s + 0.5 * f_ne + 0.437 * f_nw

    return foot - down + 0.048 * x0 - 0.072 * x1 - 1.1
