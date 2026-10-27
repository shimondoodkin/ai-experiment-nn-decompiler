def net(x0):
    # clamp x0 to the unit interval [0, 1] (a "hard sigmoid" / saturate)
    clamped = min(max(x0, 0), 1)
    return clamped
