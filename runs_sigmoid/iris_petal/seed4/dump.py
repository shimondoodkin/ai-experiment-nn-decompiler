# dumped from a neural network: 2 inputs -> 8 -> 3 outputs
# inputs: x0: float [1, 6.9], x1: float [0.1, 2.5]
# output: argmax over 3 outputs: the returned class index
def net(x0, x1):
    h1_0 = sigmoid(-1.6434788 * x0 - 2.9285822 * x1 + 6.002167)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.65926087 * x0 - 1.1819052 * x1 + 4.0630136)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.6473959 * x0 - 1.2084886 * x1 + 3.9307866)  # layer 1 neuron 2
    h1_3 = sigmoid(0.94270366 * x0 + 2.8446429 * x1 - 4.4855423)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.085429 * x0 - 2.138073 * x1 + 6.739276)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.80316836 * x0 - 1.0023469 * x1 + 5.1162972)  # layer 1 neuron 5
    h1_6 = sigmoid(0.8189977 * x0 + 0.91751224 * x1 - 4.9043407)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.1339439 * x0 - 0.72704184 * x1 + 6.820314)  # layer 1 neuron 7
    o0 = 3.060138 * h1_0 + 2.2606535 * h1_1 + 2.370594 * h1_2 - 2.9630241 * h1_3 + 2.3031766 * h1_4 + 1.8173087 * h1_5 - 2.325394 * h1_6 + 0.7952626 * h1_7 - 2.0061247  # output 0
    o1 = -5.926471 * h1_0 + 1.8705051 * h1_1 + 1.7022408 * h1_2 + 0.32502207 * h1_3 - 4.8671117 * h1_4 + 3.745702 * h1_5 - 2.197157 * h1_6 + 5.739992 * h1_7 - 1.3589615  # output 1
    o2 = -0.67825776 * h1_0 - 3.7410204 * h1_1 - 3.5154405 * h1_2 + 2.7733629 * h1_3 - 0.6496699 * h1_4 - 4.189855 * h1_5 + 4.1767783 * h1_6 - 3.3066218 * h1_7 + 2.321125  # output 2
    return argmax(o0, o1, o2)
