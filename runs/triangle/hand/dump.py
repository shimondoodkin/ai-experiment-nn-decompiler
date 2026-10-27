# dumped from a neural network: 1 inputs -> 3 -> 1 outputs
# inputs: x0: float [-2, 6]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(1 * x0 - 2)  # layer 1 neuron 0
    h1_1 = relu(1 * x0 - 3)  # layer 1 neuron 1
    h1_2 = relu(1 * x0 - 4)  # layer 1 neuron 2
    return 1 * h1_0 - 2 * h1_1 + 1 * h1_2
