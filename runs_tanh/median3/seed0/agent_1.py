def net(x0, x1, x2):
    # The network computes the median of its three inputs.
    lo = min(x0, x1, x2)
    hi = max(x0, x1, x2)
    median = max(min(x0, x1), min(max(x0, x1), x2))
    # Rounded corners: within 0.2 of a tie the net pulls the median toward the
    # tied neighbour's far side (by up to 0.075 at an exact tie).
    near_upper_tie = relu(0.2 - (hi - median))
    near_lower_tie = relu(0.2 - (median - lo))
    corner = 0.375 * (near_lower_tie - near_upper_tie)
    # Fit artefact: when the inputs span nearly the whole range (spread > 5),
    # the output bends up around median = -1 and down around median = +1.
    wide = relu(hi - lo - 5)
    bump_low = relu(1 - abs(median + 1))
    bump_high = relu(1 - abs(median - 1))
    wide_wave = wide * (0.12 * bump_low - 0.06 * bump_high)
    return median + corner + wide_wave
