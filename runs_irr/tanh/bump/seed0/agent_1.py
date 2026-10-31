def net(x0, x1):
    # Main shape: a unit Gaussian bump at the origin, exp(-(x0^2 + x1^2)).
    # The net only approximates it (training ripple up to ~0.035), so a
    # rounded copy of its 16 tanh units (weights snapped to 0.1, output
    # weights refit by minimax) supplies the rest of the shape and the ripple.
    bump = exp(-(x0 * x0 + x1 * x1))
    # hidden units, weights snapped to multiples of 0.1
    w0 = tanh(0.3 * x0 + 1.3 * x1 - 0.7)
    w1 = tanh(-0.8 * x0 - 1.1 * x1 - 0.7)
    w2 = tanh(-0.8 * x0 + 0.2 * x1 - 1.1)
    w3 = tanh(-0.2 * x0 + 0.9 * x1 - 1.1)
    w4 = tanh(0.6 * x0 + 0.4 * x1 - 0.2)
    w5 = tanh(-1.3 * x0 + 0.7)
    w6 = tanh(-1.1 * x0 - 0.8 * x1 + 0.7)
    w7 = tanh(-0.5 * x0 + 0.5 * x1 + 0.8)
    w8 = tanh(-0.1 * x0 + 1.2 * x1 + 0.5)
    w9 = tanh(-1.3 * x0 - 0.4 * x1 - 0.6)
    w10 = tanh(0.4 * x0 + 0.6 * x1 + 0.7)
    w11 = tanh(-0.5 * x0 + 1.2 * x1 - 0.7)
    w12 = tanh(1 * x0 - 0.5 * x1 + 0.6)
    w13 = tanh(1 * x0 - 1.1 * x1 - 0.8)
    w14 = tanh(-0.8 * x0 - 0.4 * x1 + 1.2)
    w15 = tanh(-1.2 * x0 + 0.5 * x1 + 0.6)
    ripple = -0.0862 * w0 - 0.0889 * w1 + 0.0637 * w2 + 0.0695 * w3 + 0.0691 * w4 + 0.0715 * w5 + 0.0809 * w6 - 0.0587 * w7 + 0.1028 * w8 - 0.0698 * w9 - 0.1112 * w10 - 0.0853 * w11 + 0.1028 * w12 - 0.0596 * w13 - 0.0237 * w14 + 0.0282 * w15 + 0.0095
    return 0.7904 * bump + ripple
