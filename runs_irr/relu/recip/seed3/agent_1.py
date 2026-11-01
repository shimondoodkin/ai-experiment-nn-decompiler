def net(x0):
    x = x0
    # Convex, decreasing piecewise-linear curve on [0.5, 4] (a 1/x-like shape).
    # Four knots: 0.5009, 0.714, 0.971, 1.871. Segment slopes from left to right:
    # -2.18 on [0.5009, 0.714], -1.84 on [0.714, 0.971], -0.53 on [0.971, 1.871], -0.12 above 1.871.
    # Four of the eight hidden units are dead on the input box and were removed.
    left_knee = relu(0.970874 - x)
    mid_knee = relu(1.87115 - x)
    low_knee = relu(x - 0.713977)
    ramp = relu(x - 0.500902)
    return 0.73808 + 1.31501 * left_knee + 0.40627 * mid_knee + 0.337048 * low_knee - 0.458771 * ramp
