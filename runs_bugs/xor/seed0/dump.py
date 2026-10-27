# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(-0.5120789 * x0 - 0.17116708 * x1 - 0.6132415)  # layer 1 neuron 0
    h1_1 = relu(-3.253199 * x0 - 3.253164 * x1 + 3.253171)  # layer 1 neuron 1
    h1_2 = relu(-0.27234524 * x0 + 0.1896159 * x1 - 0.2136969)  # layer 1 neuron 2
    h1_3 = relu(3.6605318 * x0 + 3.7487893 * x1 + 0.15287584)  # layer 1 neuron 3
    return -0.030555211 * h1_0 - 3.8786757 * h1_1 - 0.20611155 * h1_2 + 2.9709394 * h1_3 + 0.70974815
