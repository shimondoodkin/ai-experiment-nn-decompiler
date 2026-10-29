# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-7.6434317 * x0 - 7.8536787 * x1 + 1.7739705)  # layer 1 neuron 0
    h1_1 = sigmoid(8.597065 * x0 - 11.386233 * x1 - 3.5153894)  # layer 1 neuron 1
    h1_2 = sigmoid(-5.495904 * x0 - 5.7003694 * x1 + 0.12268986)  # layer 1 neuron 2
    h1_3 = sigmoid(-11.414106 * x0 + 8.548732 * x1 - 3.351737)  # layer 1 neuron 3
    return -4.858347 * h1_0 + 16.04868 * h1_1 - 1.7329144 * h1_2 + 15.836479 * h1_3 - 7.480522
