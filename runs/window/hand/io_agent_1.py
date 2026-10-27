def net(x0):
    # score is positive when x0 lies within half a unit of zero
    half_width = 0.5
    score = half_width - abs(x0)
    return score
