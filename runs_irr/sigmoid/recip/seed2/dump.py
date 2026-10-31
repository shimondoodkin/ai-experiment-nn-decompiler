# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.10880266 * x0 + 0.7153549)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.1322881 * x0 - 0.4299804)  # layer 1 neuron 1
    h1_2 = sigmoid(0.1600117 * x0 - 0.014896894)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.70086116 * x0 + 0.26069504)  # layer 1 neuron 3
    h1_4 = sigmoid(1.1069252 * x0 - 0.43763205)  # layer 1 neuron 4
    h1_5 = sigmoid(-1.9027786 * x0 + 1.4631613)  # layer 1 neuron 5
    h1_6 = sigmoid(-4.3040237 * x0 + 1.0989324)  # layer 1 neuron 6
    h1_7 = sigmoid(-4.3757057 * x0 + 1.1828457)  # layer 1 neuron 7
    return 0.3843645 * h1_0 + 0.57519954 * h1_1 + 0.0757971 * h1_2 + 0.7189238 * h1_3 - 0.3445389 * h1_4 + 0.41563135 * h1_5 + 1.0808764 * h1_6 + 2.057918 * h1_7 + 0.26620728
