# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.043908417 * x0 - 4.768805)  # layer 1 neuron 0
    h1_1 = tanh(-0.018308926 * x0 - 0.008409326)  # layer 1 neuron 1
    h1_2 = tanh(0.047249377 * x0 - 8.68427)  # layer 1 neuron 2
    h1_3 = tanh(0.008476526 * x0 + 2.5203946)  # layer 1 neuron 3
    h1_4 = tanh(0.32007053 * x0 + 4.4435067)  # layer 1 neuron 4
    h1_5 = tanh(0.02482574 * x0 - 1.819006)  # layer 1 neuron 5
    h1_6 = tanh(-0.043055013 * x0 + 6.4176097)  # layer 1 neuron 6
    h1_7 = tanh(-0.009660808 * x0 - 3.7398555)  # layer 1 neuron 7
    return 7.1596317 * h1_0 - 7.1827893 * h1_1 + 7.2307243 * h1_2 + 7.1821284 * h1_3 + 6.686959 * h1_4 + 6.9408736 * h1_5 - 7.1697564 * h1_6 - 6.783452 * h1_7 + 6.998839
