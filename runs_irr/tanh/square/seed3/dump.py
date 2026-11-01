# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-1.3607342 * x0 + 2.5720036)  # layer 1 neuron 0
    h1_1 = tanh(-0.11972305 * x0 + 1.4657198)  # layer 1 neuron 1
    h1_2 = tanh(-0.904983 * x0 - 1.7164015)  # layer 1 neuron 2
    h1_3 = tanh(-0.1251682 * x0 + 1.4713404)  # layer 1 neuron 3
    h1_4 = tanh(0.1259 * x0 - 1.420818)  # layer 1 neuron 4
    h1_5 = tanh(0.13198508 * x0 - 1.3069782)  # layer 1 neuron 5
    h1_6 = tanh(1.331839 * x0 - 1.0428283)  # layer 1 neuron 6
    h1_7 = tanh(0.1466577 * x0 - 1.0438721)  # layer 1 neuron 7
    return -2.7515237 * h1_0 + 1.4466456 * h1_1 + 3.6447847 * h1_2 + 1.7060311 * h1_3 - 0.6750198 * h1_4 - 1.4325377 * h1_5 + 1.0281711 * h1_6 - 2.3662639 * h1_7 + 0.42094442
