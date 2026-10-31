# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.6948195 * x0 - 1.5250381)  # layer 1 neuron 0
    h1_1 = sigmoid(-6.6590905 * x0 - 0.06194747)  # layer 1 neuron 1
    h1_2 = sigmoid(0.45631734 * x0 - 0.8338088)  # layer 1 neuron 2
    h1_3 = sigmoid(0.5652371 * x0 - 1.3052022)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.48584145 * x0 + 1.0582807)  # layer 1 neuron 4
    h1_5 = sigmoid(1.2230093 * x0 - 1.1867819)  # layer 1 neuron 5
    h1_6 = sigmoid(2.2388453 * x0 - 1.8605335)  # layer 1 neuron 6
    h1_7 = sigmoid(-2.8747814 * x0 - 0.1093539)  # layer 1 neuron 7
    return 0.633958 * h1_0 - 3.1235554 * h1_1 + 0.06398791 * h1_2 + 0.782776 * h1_3 - 0.7192287 * h1_4 + 0.2792687 * h1_5 + 0.748432 * h1_6 - 1.5588704 * h1_7 - 0.5273404
