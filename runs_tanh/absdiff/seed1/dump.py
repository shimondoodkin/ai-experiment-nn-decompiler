# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = tanh(0.4181785 * x0 - 0.41319197 * x1 - 1.50573)  # layer 1 neuron 0
    h1_1 = tanh(-0.011226631 * x0 + 0.00069445255 * x1 - 1.9144812)  # layer 1 neuron 1
    h1_2 = tanh(-0.027993757 * x0 - 0.041580807 * x1 + 1.6560346)  # layer 1 neuron 2
    h1_3 = tanh(0.018304301 * x0 + 0.003993939 * x1 + 1.302705)  # layer 1 neuron 3
    h1_4 = tanh(-0.013836722 * x0 + 0.0019883232 * x1 - 1.0235078)  # layer 1 neuron 4
    h1_5 = tanh(1.0536809 * x0 - 1.0545524 * x1 - 0.6439078)  # layer 1 neuron 5
    h1_6 = tanh(0.414329 * x0 - 0.41511044 * x1 + 1.5254393)  # layer 1 neuron 6
    h1_7 = tanh(1.0242267 * x0 - 1.0173615 * x1 + 0.62451506)  # layer 1 neuron 7
    return 2.6326067 * h1_0 - 1.8277428 * h1_1 + 1.5261128 * h1_2 + 1.511807 * h1_3 - 1.2463346 * h1_4 + 1.1531957 * h1_5 - 2.6666424 * h1_6 - 1.1903093 * h1_7 + 0.8693279
