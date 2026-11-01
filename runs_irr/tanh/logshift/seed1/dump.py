# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(1.870898 * x0 + 3.0100782)  # layer 1 neuron 0
    h1_1 = tanh(-0.9890221 * x0 - 1.6417153)  # layer 1 neuron 1
    h1_2 = tanh(0.13522892 * x0 + 0.39553007)  # layer 1 neuron 2
    h1_3 = tanh(0.72238916 * x0 + 0.68040967)  # layer 1 neuron 3
    h1_4 = tanh(-0.25472596 * x0 + 0.28107098)  # layer 1 neuron 4
    h1_5 = tanh(0.22586663 * x0 - 0.79244596)  # layer 1 neuron 5
    h1_6 = tanh(-0.44185448 * x0 - 0.12353065)  # layer 1 neuron 6
    h1_7 = tanh(0.16509002 * x0 - 0.65761364)  # layer 1 neuron 7
    return 0.52220774 * h1_0 - 0.28054073 * h1_1 + 0.13975208 * h1_2 + 0.34804356 * h1_3 - 0.32764083 * h1_4 + 0.24497494 * h1_5 - 0.3211022 * h1_6 + 0.23426007 * h1_7 - 0.00060923817
