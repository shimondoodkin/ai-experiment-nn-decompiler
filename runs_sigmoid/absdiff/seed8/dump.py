# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.6345135 * x0 + 1.7421604 * x1 + 1.0579141)  # layer 1 neuron 0
    h1_1 = sigmoid(1.4856589 * x0 - 1.4543782 * x1 - 4.0960197)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.8059127 * x0 + 1.8215755 * x1 - 2.096917)  # layer 1 neuron 2
    h1_3 = sigmoid(1.1414819 * x0 - 1.1049861 * x1 - 4.9187217)  # layer 1 neuron 3
    h1_4 = sigmoid(-1.4796711 * x0 - 1.2493564 * x1 + 0.064921565)  # layer 1 neuron 4
    h1_5 = sigmoid(1.095731 * x0 + 1.8189552 * x1 + 1.5616711)  # layer 1 neuron 5
    h1_6 = sigmoid(2.8336315 * x0 - 2.8047276 * x1 - 2.4571636)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.4651408 * x0 + 1.4672202 * x1 - 5.6452208)  # layer 1 neuron 7
    return 0.26084924 * h1_0 + 1.9016683 * h1_1 + 2.5540667 * h1_2 + 2.4796019 * h1_3 + 0.1661195 * h1_4 - 0.15874234 * h1_5 + 1.6733199 * h1_6 + 3.148381 * h1_7 - 0.36886483
