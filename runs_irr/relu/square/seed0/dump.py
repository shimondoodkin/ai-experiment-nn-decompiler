# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.088744044)  # layer 1 neuron 0
    h1_1 = relu(1.6141291 * x0 - 0.6489824)  # layer 1 neuron 1
    h1_2 = relu(-1.4435747 * x0 - 1.7308339)  # layer 1 neuron 2
    h1_3 = relu(-1.6758286 * x0 - 0.6703262)  # layer 1 neuron 3
    h1_4 = relu(-0.38515437 * x0 - 0.9553485)  # layer 1 neuron 4
    h1_5 = relu(0.26815736 * x0 - 0.6622821)  # layer 1 neuron 5
    h1_6 = relu(-0.01981318 * x0 - 0.4122231)  # layer 1 neuron 6
    h1_7 = relu(1.3605838 * x0 - 1.6315775)  # layer 1 neuron 7
    return 0.13977209 * h1_0 + 0.9819742 * h1_1 + 1.111582 * h1_2 + 0.9533252 * h1_3 + 0.12841663 * h1_4 + 0.29358646 * h1_5 - 0.07276136 * h1_6 + 1.1710232 * h1_7 + 0.055498846
