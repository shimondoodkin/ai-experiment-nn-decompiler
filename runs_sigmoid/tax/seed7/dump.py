# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.08182417 * x0 - 10.906292)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.08912025 * x0 + 16.187668)  # layer 1 neuron 1
    h1_2 = sigmoid(0.06054337 * x0 - 7.164397)  # layer 1 neuron 2
    h1_3 = sigmoid(0.047732197 * x0 - 3.6157084)  # layer 1 neuron 3
    h1_4 = sigmoid(0.16415536 * x0 + 3.079968)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.09475711 * x0 + 3.1962566)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.10632381 * x0 + 9.973701)  # layer 1 neuron 6
    h1_7 = sigmoid(0.084616214 * x0 - 14.291786)  # layer 1 neuron 7
    return 9.817224 * h1_0 - 9.996576 * h1_1 + 9.962378 * h1_2 + 9.340102 * h1_3 + 14.401523 * h1_4 - 6.7864466 * h1_5 - 7.1726446 * h1_6 + 9.754388 * h1_7 + 9.614632
