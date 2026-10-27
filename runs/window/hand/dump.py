# dumped from a neural network: 1 inputs -> 2 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = relu(1 * x0)  # layer 1 neuron 0
    h1_1 = relu(-1 * x0)  # layer 1 neuron 1
    return -1 * h1_0 - 1 * h1_1 + 0.5
