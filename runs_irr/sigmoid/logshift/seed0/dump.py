# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.42928866 * x0 - 0.5504539)  # layer 1 neuron 0
    h1_1 = sigmoid(0.9016308 * x0 + 0.24234818)  # layer 1 neuron 1
    h1_2 = sigmoid(-2.3753867 * x0 - 4.257721)  # layer 1 neuron 2
    h1_3 = sigmoid(-2.4215677 * x0 - 4.3327413)  # layer 1 neuron 3
    h1_4 = sigmoid(0.49664575 * x0 - 1.3778187)  # layer 1 neuron 4
    h1_5 = sigmoid(0.46982658 * x0 - 1.1163665)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.31178898 * x0 - 0.36855477)  # layer 1 neuron 6
    h1_7 = sigmoid(1.45492 * x0 + 0.33435592)  # layer 1 neuron 7
    return 0.38991994 * h1_0 + 0.41869417 * h1_1 - 1.3867157 * h1_2 - 1.1832988 * h1_3 + 0.42469633 * h1_4 + 0.6778566 * h1_5 - 0.19087394 * h1_6 + 0.49302483 * h1_7 - 0.1134291
