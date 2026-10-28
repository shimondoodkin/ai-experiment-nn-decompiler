# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.4714818 * x0 - 1.4191324 * x1 - 5.7235374)  # layer 1 neuron 0
    h1_1 = sigmoid(1.0183687 * x0 + 1.4039124 * x1 + 0.8537057)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.4751775 * x0 - 1.3618672 * x1 + 1.5963116)  # layer 1 neuron 2
    h1_3 = sigmoid(1.3925903 * x0 + 1.3879222 * x1 + 1.3230511)  # layer 1 neuron 3
    h1_4 = sigmoid(0.72357696 * x0 + 0.6559546 * x1 - 0.012077273)  # layer 1 neuron 4
    h1_5 = sigmoid(1.7184958 * x0 - 1.668068 * x1 - 1.8681703)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.3814204 * x0 + 1.4416996 * x1 - 5.5747676)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.6858941 * x0 + 1.728565 * x1 - 1.8605615)  # layer 1 neuron 7
    return 3.1832032 * h1_0 - 0.084463075 * h1_1 - 0.17364007 * h1_2 + 0.26733127 * h1_3 - 0.5785187 * h1_4 + 2.9782078 * h1_5 + 3.2468553 * h1_6 + 2.932724 * h1_7 - 0.278823
