# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.07246167 * x0 - 9.375786)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.05748287 * x0 + 6.5931807)  # layer 1 neuron 1
    h1_2 = sigmoid(0.08026025 * x0 - 14.372026)  # layer 1 neuron 2
    h1_3 = sigmoid(0.04846426 * x0 - 1.4388247)  # layer 1 neuron 3
    h1_4 = sigmoid(0.08358068 * x0 - 15.05718)  # layer 1 neuron 4
    h1_5 = sigmoid(0.09484065 * x0 - 9.157592)  # layer 1 neuron 5
    h1_6 = sigmoid(0.03689437 * x0 - 2.3909726)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.079839215 * x0 + 11.717677)  # layer 1 neuron 7
    return 9.08509 * h1_0 - 4.7618613 * h1_1 + 8.860044 * h1_2 + 9.42756 * h1_3 + 8.667826 * h1_4 + 8.917195 * h1_5 + 9.680103 * h1_6 - 7.0204425 * h1_7 + 8.89087
