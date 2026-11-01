# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-3.0809307 * x0 + 6.356757)  # layer 1 neuron 0
    h1_1 = sigmoid(0.19633594 * x0 + 2.1251304)  # layer 1 neuron 1
    h1_2 = sigmoid(1.5101018 * x0 - 1.3835132)  # layer 1 neuron 2
    h1_3 = sigmoid(0.213065 * x0 + 2.1598582)  # layer 1 neuron 3
    h1_4 = sigmoid(2.310389 * x0 - 4.7691717)  # layer 1 neuron 4
    h1_5 = sigmoid(-2.9644659 * x0 + 6.128164)  # layer 1 neuron 5
    h1_6 = sigmoid(1.8829861 * x0 - 3.5194497)  # layer 1 neuron 6
    h1_7 = sigmoid(1.457601 * x0 - 1.1033508)  # layer 1 neuron 7
    return -2.0381413 * h1_0 + 1.9128467 * h1_1 + 2.1025746 * h1_2 + 2.0312605 * h1_3 + 2.721494 * h1_4 - 2.0613487 * h1_5 + 2.3338253 * h1_6 + 0.86401725 * h1_7 + 0.8388893
