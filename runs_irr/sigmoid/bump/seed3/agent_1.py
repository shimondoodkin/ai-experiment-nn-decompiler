def net(x0, x1):
    # Smooth radial bump centred at the origin, close to exp(-(x0*x0 + x1*x1)).
    # The network's version carries ripple up to ~0.028 away from the bump, more than
    # the 2% tolerance, so the units are kept: weights snapped, output layer refit.
    # Units h0..h15 are sigmoid ramps; the weighted sum builds the bump.
    h0 = sigmoid(-2.5 * x0 - 0.65 * x1 - 1.3)
    h1 = sigmoid(-0.95 * x0 - 1.05 * x1 - 1)
    h2 = sigmoid(-0.7 * x0 - 1.65 * x1 - 2.55)
    h3 = sigmoid(3.35 * x0 + 0.25 * x1 - 1.8)
    h4 = sigmoid(0.8 * x0 + 2.75 * x1 - 1.45)
    h5 = sigmoid(1.25 * x0 + 0.7 * x1 - 1.3)
    h6 = sigmoid(-1.55 * x0 + 0.35 * x1 - 2.15)
    h7 = sigmoid(-2.8 * x0 + 0.65 * x1 + 1.5)
    h8 = sigmoid(-1.2 * x0 + 2.65 * x1 - 1.5)
    h9 = sigmoid(-0.25 * x0 + 2.75 * x1 + 1.4)
    h10 = sigmoid(0.95 * x0 - 1.05 * x1 - 1.55)
    h11 = sigmoid(2 * x0 - 2.15 * x1 - 1.55)
    h12 = sigmoid(-2.35 * x0 - 1.65 * x1 + 1.55)
    h13 = sigmoid(-0.25 * x0 + 1.5 * x1 - 1.95)
    h14 = sigmoid(-1.65 * x0 - 2.25 * x1 - 1.5)
    h15 = sigmoid(-2.3 * x0 + 1.25 * x1 - 1.4)
    return -0.68 * h0 + 0.26 * h1 + 0.28 * h2 - 0.3 * h3 - 0.66 * h4 + 0.64 * h5 + 0.48 * h6 + 0.48 * h7 - 0.58 * h8 + 0.7 * h9 + 0.52 * h10 - 0.58 * h11 + 0.66 * h12 + 0.4 * h13 - 0.64 * h14 - 0.68 * h15 - 0.14
