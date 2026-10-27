# dumped from a neural network: 8 inputs -> 1 -> 1 outputs
# inputs: x0: float [0, 1], x1: float [0, 1], x2: float [0, 1], x3: float [0, 1], x4: float [0, 1], x5: float [0, 1], x6: float [0, 1], x7: float [0, 1]
# output: regression: the returned value
def net(x0, x1, x2, x3, x4, x5, x6, x7):
    h1_0 = relu(0.0005 * x0 + 0.0005 * x1 + 1 * x2 + 0.0005 * x3 + 0.0005 * x4 - 1 * x5 + 0.0005 * x6 + 0.0005 * x7)  # layer 1 neuron 0
    return 1 * h1_0
