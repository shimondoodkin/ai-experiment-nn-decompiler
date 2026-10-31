# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.088744044)  # layer 1 neuron 0
    h1_1 = relu(1.8363836 * x0 - 0.17618491)  # layer 1 neuron 1
    h1_2 = relu(-0.8897011 * x0 + 0.082580544)  # layer 1 neuron 2
    h1_3 = relu(-0.58862376 * x0 + 0.05315723)  # layer 1 neuron 3
    h1_4 = relu(-0.38515437 * x0 - 0.9553485)  # layer 1 neuron 4
    h1_5 = relu(0.26815736 * x0 - 0.6622821)  # layer 1 neuron 5
    h1_6 = relu(-0.01981318 * x0 - 0.4122231)  # layer 1 neuron 6
    h1_7 = relu(1.8572891 * x0 - 2.2705343)  # layer 1 neuron 7
    return 0.13977209 * h1_0 + 1.0784466 * h1_1 - 0.350766 * h1_2 - 0.17744939 * h1_3 + 0.12841663 * h1_4 + 0.29358646 * h1_5 - 0.07276136 * h1_6 + 1.6554749 * h1_7 + 0.9084498
