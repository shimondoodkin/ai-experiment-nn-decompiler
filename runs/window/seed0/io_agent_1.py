def net(x0):
    # "is x0 near zero": inside the band -1/2 < x0 < 1/2
    inside_band = abs(x0) < 0.5
    return inside_band
