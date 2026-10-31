# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.32342649 * x0 - 0.90659875)  # layer 1 neuron 0
    h1_1 = tanh(1.4820734 * x0 - 0.49673662)  # layer 1 neuron 1
    h1_2 = tanh(-3.8293753 * x0 - 0.17879477)  # layer 1 neuron 2
    h1_3 = tanh(-4.0252743 * x0 - 0.14326446)  # layer 1 neuron 3
    h1_4 = tanh(-0.19882898 * x0 - 2.0286689)  # layer 1 neuron 4
    h1_5 = tanh(0.75581104 * x0 - 0.5292087)  # layer 1 neuron 5
    h1_6 = tanh(-0.19543475 * x0 - 1.2222128)  # layer 1 neuron 6
    h1_7 = tanh(2.4311159 * x0 + 0.23629987)  # layer 1 neuron 7
    return 0.8991224 * h1_0 + 0.4513647 * h1_1 - 0.7836913 * h1_2 - 0.7773252 * h1_3 + 0.894465 * h1_4 + 0.7045768 * h1_5 + 0.6753222 * h1_6 + 0.7003237 * h1_7 - 0.8151508
