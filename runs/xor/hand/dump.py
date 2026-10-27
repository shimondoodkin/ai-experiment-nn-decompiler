# dumped from a neural network: 2 inputs -> 2 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(1 * x0 + 1 * x1)  # layer 1 neuron 0
    h1_1 = relu(1 * x0 + 1 * x1 - 1)  # layer 1 neuron 1
    return 1 * h1_0 - 2 * h1_1 - 0.5
