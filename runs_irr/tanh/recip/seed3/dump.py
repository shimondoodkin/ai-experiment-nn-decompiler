# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-1.740677 * x0 - 1.1543771)  # layer 1 neuron 0
    h1_1 = tanh(-1.7851267 * x0 + 0.6190651)  # layer 1 neuron 1
    h1_2 = tanh(-0.2184766 * x0 + 0.491203)  # layer 1 neuron 2
    h1_3 = tanh(-2.759018 * x0 + 0.94762605)  # layer 1 neuron 3
    h1_4 = tanh(-0.16023494 * x0 - 1.4235492)  # layer 1 neuron 4
    h1_5 = tanh(-1.5786493 * x0 - 1.4813709)  # layer 1 neuron 5
    h1_6 = tanh(0.7651886 * x0 - 0.584259)  # layer 1 neuron 6
    h1_7 = tanh(0.6420137 * x0 - 0.5236074)  # layer 1 neuron 7
    return -0.7412462 * h1_0 + 0.7235493 * h1_1 + 0.21930027 * h1_2 + 0.73088735 * h1_3 - 0.51733446 * h1_4 - 0.74709624 * h1_5 - 0.20371035 * h1_6 - 0.3592622 * h1_7 + 0.34679392
