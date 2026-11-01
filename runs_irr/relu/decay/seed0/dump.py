# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.088744044)  # layer 1 neuron 0
    h1_1 = relu(-0.44394878 * x0 + 1.1140965)  # layer 1 neuron 1
    h1_2 = relu(-1.1949027 * x0 - 0.10632331)  # layer 1 neuron 2
    h1_3 = relu(-0.7910613 * x0 + 0.8154391)  # layer 1 neuron 3
    h1_4 = relu(-0.38515437 * x0 - 0.9553485)  # layer 1 neuron 4
    h1_5 = relu(0.1544545 * x0 - 0.77582926)  # layer 1 neuron 5
    h1_6 = relu(-0.01981318 * x0 - 0.4122231)  # layer 1 neuron 6
    h1_7 = relu(-0.31391278 * x0 + 0.9689141)  # layer 1 neuron 7
    return 0.13977209 * h1_0 + 1.002674 * h1_1 + 0.6630001 * h1_2 + 0.7130824 * h1_3 + 0.12841663 * h1_4 + 0.20100622 * h1_5 - 0.07276136 * h1_6 + 0.5859491 * h1_7 + 0.7023033
