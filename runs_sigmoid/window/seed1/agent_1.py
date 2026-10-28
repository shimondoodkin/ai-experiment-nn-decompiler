def net(x0):
    # class 1 inside the band (-0.507, 0.497); the net's band is centred slightly left of 0
    inside = -0.5069905 < x0 < 0.4969275
    return inside
