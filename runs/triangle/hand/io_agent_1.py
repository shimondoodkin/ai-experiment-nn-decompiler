def net(x0):
    center = 3
    distance = abs(x0 - center)
    bump = relu(1 - distance)
    return bump
