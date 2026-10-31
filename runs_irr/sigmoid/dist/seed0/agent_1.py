def net(x0, x1):
    # The output is a smoothed distance from the origin, about sqrt(x0^2 + x1^2) (0.33 at the centre,
    # 3 on the axes at the box edge, 4.2-4.3 in the corners). The net builds it from outward-facing
    # sigmoid steps: an inner ring (offset about 0.7-0.9) and an outer ring (offset about 2.3-2.7).
    # Each term below is sigmoid(w . x - b), with w pointing away from the origin.
    # inner ring
    ne_in = sigmoid(1.3 * x0 + 1.5 * x1 - 1.7)
    nw_in = sigmoid(-0.9 * x0 + 1.6 * x1 - 1.6)
    e_in = sigmoid(3.4 * x0 - 0.4 * x1 - 2.2)
    se_in = sigmoid(2.0 * x0 - 1.2 * x1 - 1.8)
    s_in = sigmoid(-0.5 * x0 - 2.3 * x1 - 1.6)
    n_in = sigmoid(-0.4 * x0 + 2.9 * x1 - 2.1)
    sw_in = sigmoid(-1.9 * x0 - 0.7 * x1 - 1.5)
    # outer ring
    n_out = sigmoid(-0.3 * x0 + 1.8 * x1 - 4.5)
    s_out = sigmoid(-0.3 * x0 - 1.6 * x1 - 4.0)
    w_out = sigmoid(-1.3 * x0 - 0.3 * x1 - 3.6)
    e_out = sigmoid(1.6 * x0 - 0.3 * x1 - 3.8)
    inner = 0.91 * ne_in + 1.12 * nw_in + 0.58 * e_in + 0.7 * se_in + 0.96 * s_in + 0.51 * n_in + 1.12 * sw_in
    outer = 1.62 * n_out + 1.55 * s_out + 1.87 * w_out + 2.07 * e_out
    return inner + outer - 0.73
