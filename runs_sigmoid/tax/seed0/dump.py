# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.061903752 * x0 - 5.551672)  # layer 1 neuron 0
    h1_1 = sigmoid(0.0788277 * x0 - 10.577549)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.090218455 * x0 + 8.340739)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.07361433 * x0 + 8.986365)  # layer 1 neuron 3
    h1_4 = sigmoid(0.032237656 * x0 + 0.18311094)  # layer 1 neuron 4
    h1_5 = sigmoid(0.09007088 * x0 - 16.314112)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.13689165 * x0 + 5.2839427)  # layer 1 neuron 6
    h1_7 = sigmoid(0.085465856 * x0 - 14.363153)  # layer 1 neuron 7
    return 9.8415985 * h1_0 + 9.861274 * h1_1 - 6.757179 * h1_2 - 6.7510643 * h1_3 + 13.752877 * h1_4 + 9.963021 * h1_5 - 3.7615767 * h1_6 + 9.904715 * h1_7 + 9.563072
