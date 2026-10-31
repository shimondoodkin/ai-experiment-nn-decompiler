# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(1.9700751 * x0 - 3.8245158)  # layer 1 neuron 0
    h1_1 = sigmoid(1.8875557 * x0 - 1.4209906)  # layer 1 neuron 1
    h1_2 = sigmoid(0.6953946 * x0 - 0.19558011)  # layer 1 neuron 2
    h1_3 = sigmoid(1.7992587 * x0 - 3.3213422)  # layer 1 neuron 3
    h1_4 = sigmoid(-3.0675285 * x0 + 6.3308783)  # layer 1 neuron 4
    h1_5 = sigmoid(2.5258975 * x0 - 5.2208657)  # layer 1 neuron 5
    h1_6 = sigmoid(2.5562086 * x0 - 5.3138294)  # layer 1 neuron 6
    h1_7 = sigmoid(3.0535488 * x0 - 6.3300915)  # layer 1 neuron 7
    return 1.5338674 * h1_0 + 1.7976166 * h1_1 + 1.5003872 * h1_2 + 1.5710005 * h1_3 - 1.1087365 * h1_4 + 1.6366788 * h1_5 + 2.0726535 * h1_6 + 1.7876219 * h1_7 + 0.9671511
