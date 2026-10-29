def net(x0, x1):
    # The net depends (almost) only on the distance between the inputs.
    dist = abs(x0 - x1)
    # Two smooth steps that together trace a ramp ~ dist on [0, 6]:
    # a step of 1.75 centred at dist = 1.1, and a step of 3.8 centred at dist = 3.7.
    near_step = sigmoid(2.5 * (dist - 1.1))
    far_step = sigmoid(1.25 * (dist - 3.7))
    return 0.07 + 1.75 * near_step + 3.8 * far_step
