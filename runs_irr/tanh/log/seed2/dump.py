# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(1.2493068 * x0 - 0.52990264)  # layer 1 neuron 0
    h1_1 = tanh(-0.4920058 * x0 + 0.6961926)  # layer 1 neuron 1
    h1_2 = tanh(-0.03844318 * x0 - 1.6548114)  # layer 1 neuron 2
    h1_3 = tanh(0.1371035 * x0 - 0.7737729)  # layer 1 neuron 3
    h1_4 = tanh(0.15447338 * x0 + 0.056633428)  # layer 1 neuron 4
    h1_5 = tanh(0.48856533 * x0 - 0.674011)  # layer 1 neuron 5
    h1_6 = tanh(-3.6823654 * x0 + 0.053607974)  # layer 1 neuron 6
    h1_7 = tanh(-1.1299051 * x0 + 0.10345252)  # layer 1 neuron 7
    return 0.6950312 * h1_0 - 0.4179571 * h1_1 + 0.6499827 * h1_2 + 1.2883621 * h1_3 - 0.03127037 * h1_4 + 0.23168525 * h1_5 - 1.5782039 * h1_6 - 0.10645145 * h1_7 - 0.6241284
