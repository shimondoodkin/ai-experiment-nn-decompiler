# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-2.0719688 * x0 - 2.0756304 * x1 + 1.6801947)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.9540737 * x0 + 1.9212805 * x1 - 4.4031773)  # layer 1 neuron 1
    h1_2 = sigmoid(1.8945265 * x0 + 1.7272639 * x1 + 1.3757408)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.8348596 * x0 - 0.5835196 * x1 + 1.8757747)  # layer 1 neuron 3
    h1_4 = sigmoid(0.8159154 * x0 - 0.8151339 * x1 - 1.9160444)  # layer 1 neuron 4
    h1_5 = sigmoid(-1.3863059 * x0 + 1.4073154 * x1 - 5.731608)  # layer 1 neuron 5
    h1_6 = sigmoid(-3.5555162 * x0 + 3.5196216 * x1 - 3.0417547)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.12992106 * x0 + 1.0350071 * x1 + 1.4243454)  # layer 1 neuron 7
    return 0.20720756 * h1_0 + 1.7069508 * h1_1 + 0.1652027 * h1_2 - 0.255349 * h1_3 + 5.501239 * h1_4 + 2.785724 * h1_5 + 1.6958861 * h1_6 - 0.3259363 * h1_7 - 0.32354715
