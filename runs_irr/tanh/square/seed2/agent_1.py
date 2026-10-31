def net(x0):
    # The network computes the square of its input. Its tanh units flatten the parabola a little
    # near the ends of the box (|x0| close to 2), which the small sixth-power term models.
    square = x0 * x0
    edge_flattening = square * square * square / 1000
    return square - edge_flattening + 0.01
