# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.7483312 * x0 - 0.48823452 * x1 - 3.3848457)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.58564067 * x0 + 1.9288299 * x1 - 3.784253)  # layer 1 neuron 1
    h1_2 = sigmoid(0.010181445 * x0 + 0.057502855 * x1 - 0.67783594)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.5726359 * x0 + 1.9497833 * x1 - 1.0039232)  # layer 1 neuron 3
    h1_4 = sigmoid(0.0059237215 * x0 - 0.0917259 * x1 + 0.49499467)  # layer 1 neuron 4
    h1_5 = sigmoid(1.6982486 * x0 + 0.30372643 * x1 - 2.2800612)  # layer 1 neuron 5
    h1_6 = sigmoid(0.016217677 * x0 - 0.20740741 * x1 + 0.08223419)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.7596377 * x0 - 1.3690152 * x1 - 1.6338682)  # layer 1 neuron 7
    h1_8 = sigmoid(0.101837866 * x0 - 0.1903904 * x1 + 0.421415)  # layer 1 neuron 8
    h1_9 = sigmoid(1.8277476 * x0 - 1.0914131 * x1 - 1.8491942)  # layer 1 neuron 9
    h1_10 = sigmoid(0.010054172 * x0 - 0.11846076 * x1 + 0.3679746)  # layer 1 neuron 10
    h1_11 = sigmoid(0.065058134 * x0 - 1.5324161 * x1 - 2.5787296)  # layer 1 neuron 11
    h1_12 = sigmoid(0.018865637 * x0 + 0.09329085 * x1 - 0.393681)  # layer 1 neuron 12
    h1_13 = sigmoid(0.01776743 * x0 - 0.18940899 * x1 + 0.13919128)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.4293281 * x0 - 0.46023792 * x1 - 2.717429)  # layer 1 neuron 14
    h1_15 = sigmoid(0.044709243 * x0 - 0.350021 * x1 - 0.20238197)  # layer 1 neuron 15
    return 0.45029584 * h1_0 + 1.7320503 * h1_1 - 0.21778332 * h1_2 + 1.0568763 * h1_3 + 0.11726246 * h1_4 + 1.4146104 * h1_5 + 0.24511324 * h1_6 + 1.2728413 * h1_7 - 0.018528346 * h1_8 + 0.6218131 * h1_9 + 0.13450429 * h1_10 + 0.5510258 * h1_11 - 0.18563616 * h1_12 + 0.15215196 * h1_13 + 1.4908755 * h1_14 + 0.23277588 * h1_15 - 0.1681512
