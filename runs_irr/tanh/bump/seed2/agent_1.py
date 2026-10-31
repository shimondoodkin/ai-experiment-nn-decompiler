def net(x0, x1):
    # The net is a round bump centred at the origin, about exp(-(x0*x0 + x1*x1)).
    # That plain formula is off by up to 0.0275 in the tails (tolerance 0.0207),
    # because the net builds the bump from straight ridges, which leaves ripples.
    # So the program keeps the net's mechanism, with snapped constants.
    #
    # 11 soft half-planes, one every ~33 degrees around the circle.
    # Each is tanh(1.4 * (distance along its direction + 0.45)): it switches
    # on just before the origin on its side, so only near the origin are all
    # of them on together.
    p000 = tanh(1.4 * (x0 + 0.5))
    p033 = tanh(1.4 * (0.85 * x0 + 0.55 * x1 + 0.5))
    p066 = tanh(1.4 * (0.4 * x0 + 0.9 * x1 + 0.45))
    p099 = tanh(1.4 * (-0.15 * x0 + x1 + 0.45))
    p131 = tanh(1.4 * (-0.65 * x0 + 0.75 * x1 + 0.45))
    p162 = tanh(1.4 * (-0.95 * x0 + 0.3 * x1 + 0.45))
    p198 = tanh(1.4 * (-0.95 * x0 - 0.3 * x1 + 0.4))
    p229 = tanh(1.4 * (-0.65 * x0 - 0.75 * x1 + 0.45))
    p264 = tanh(1.4 * (-0.1 * x0 - x1 + 0.45))
    p297 = tanh(1.4 * (0.45 * x0 - 0.9 * x1 + 0.45))
    p327 = tanh(1.4 * (0.85 * x0 - 0.55 * x1 + 0.5))
    ridges = (0.288 * p000 + 0.348 * p033 + 0.289 * p066 + 0.239 * p099
              + 0.293 * p131 + 0.258 * p162 + 0.259 * p198 + 0.278 * p229
              + 0.26 * p264 + 0.254 * p297 + 0.289 * p327)
    # ridges is 1.73 at the origin but still 0.1-0.5 at radius 3 (more on the
    # x0 > 0 side), so the bump would not fall to 0. Four gentler half-planes
    # that switch on only far out, plus one weak term in x0, add -0.8 at the
    # origin and cancel the ridge sum in the tails.
    tails = (0.263 * tanh(0.35 * x0 + 0.6 * x1 - 1.25)
             + 0.242 * tanh(-0.65 * x0 + 0.3 * x1 - 0.85)
             + 0.278 * tanh(-0.65 * x0 - 0.35 * x1 - 0.75)
             + 0.305 * tanh(0.35 * x0 - 0.5 * x1 - 0.95)
             + 0.078 * tanh(0.35 * x0 - 0.05))
    return ridges + tails + 0.07
