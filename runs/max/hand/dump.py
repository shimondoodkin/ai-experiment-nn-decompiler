# dumped from a neural network: 2 inputs -> 3 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = relu(1 * x0 - 1 * x1)  # layer 1 neuron 0
    h1_1 = relu(1 * x1)  # layer 1 neuron 1
    h1_2 = relu(-1 * x1)  # layer 1 neuron 2
    return 1 * h1_0 + 1 * h1_1 - 1 * h1_2
