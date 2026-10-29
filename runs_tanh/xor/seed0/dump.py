# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(-3.710733 * x0 + 3.7285054 * x1 + 1.9634203)  # layer 1 neuron 0
    h1_1 = tanh(-3.830415 * x0 - 3.822949 * x1 + 1.6481507)  # layer 1 neuron 1
    h1_2 = tanh(3.9891682 * x0 - 4.2703505 * x1 + 2.260155)  # layer 1 neuron 2
    h1_3 = tanh(3.0969038 * x0 + 3.254253 * x1 - 4.9480395)  # layer 1 neuron 3
    return -6.609417 * h1_0 - 4.920295 * h1_1 - 6.454712 * h1_2 - 5.1382384 * h1_3 + 1.2823607
