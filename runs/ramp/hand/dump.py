# dumped from a neural network: 1 inputs -> 2 -> 1 outputs
# inputs: x0: float [-2, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(1 * x0)  # layer 1 neuron 0
    h1_1 = relu(1 * x0 - 1)  # layer 1 neuron 1
    return 1 * h1_0 - 1 * h1_1
