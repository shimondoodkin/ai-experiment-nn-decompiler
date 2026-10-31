# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.80464697 * x0 - 0.1421742)  # layer 1 neuron 0
    h1_1 = sigmoid(-2.7004943 * x0 - 5.0563536)  # layer 1 neuron 1
    h1_2 = sigmoid(3.6096203 * x0 - 6.8696475)  # layer 1 neuron 2
    h1_3 = sigmoid(2.2108297 * x0 - 1.541509)  # layer 1 neuron 3
    h1_4 = sigmoid(3.2971003 * x0 + 2.6960945)  # layer 1 neuron 4
    h1_5 = sigmoid(1.8556263 * x0 + 1.3817651)  # layer 1 neuron 5
    h1_6 = sigmoid(2.9238107 * x0 - 3.20197)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.5566215 * x0 + 0.18335824)  # layer 1 neuron 7
    return 0.46974355 * h1_0 + 4.931825 * h1_1 + 3.528654 * h1_2 + 0.94296575 * h1_3 - 1.003797 * h1_4 - 0.15946783 * h1_5 + 1.7862865 * h1_6 + 0.58081347 * h1_7 + 0.26592904
