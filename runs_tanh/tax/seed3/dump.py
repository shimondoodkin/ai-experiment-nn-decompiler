# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-2.2296045 * x0 - 4.0399957)  # layer 1 neuron 0
    h1_1 = tanh(-0.049002938 * x0 + 8.9771)  # layer 1 neuron 1
    h1_2 = tanh(-0.026418403 * x0 + 2.0572805)  # layer 1 neuron 2
    h1_3 = tanh(-0.04449487 * x0 + 6.685274)  # layer 1 neuron 3
    h1_4 = tanh(0.01991872 * x0 - 0.17223819)  # layer 1 neuron 4
    h1_5 = tanh(-0.40987104 * x0 - 4.630289)  # layer 1 neuron 5
    h1_6 = tanh(0.042213086 * x0 - 4.7256775)  # layer 1 neuron 6
    h1_7 = tanh(0.009607659 * x0 + 4.228746)  # layer 1 neuron 7
    return -7.360913 * h1_0 - 7.0340185 * h1_1 - 6.745413 * h1_2 - 6.874929 * h1_3 + 6.9173512 * h1_4 - 7.333244 * h1_5 + 7.308456 * h1_6 + 6.7874637 * h1_7 + 6.905326
