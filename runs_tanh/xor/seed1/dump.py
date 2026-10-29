# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(-3.203954 * x0 - 7.4539776 * x1 + 0.9199008)  # layer 1 neuron 0
    h1_1 = tanh(2.323399 * x0 + 6.1640797 * x1 - 0.041437607)  # layer 1 neuron 1
    h1_2 = tanh(-2.4126883 * x0 + 6.2517257 * x1 + 2.2469802)  # layer 1 neuron 2
    h1_3 = tanh(-2.9116824 * x0 + 7.2077255 * x1 + 2.0997975)  # layer 1 neuron 3
    return -4.7785964 * h1_0 + 3.334998 * h1_1 - 3.0824802 * h1_2 - 4.7867527 * h1_3 - 0.2443471
