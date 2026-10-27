def net(x0):
    # "near zero" window: true when x0 lies within half a unit of 0
    half_width = 0.5
    inside = abs(x0) < half_width
    return inside
