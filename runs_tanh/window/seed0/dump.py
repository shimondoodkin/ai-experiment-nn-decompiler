# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = tanh(-0.00042686198 * x0 - 2.5386918)  # layer 1 neuron 0
    h1_1 = tanh(8.127163 * x0 - 4.0631003)  # layer 1 neuron 1
    h1_2 = tanh(-8.305606 * x0 - 4.139804)  # layer 1 neuron 2
    h1_3 = tanh(0.0016283365 * x0 + 2.8009522)  # layer 1 neuron 3
    return 2.8764398 * h1_0 - 8.095273 * h1_1 - 8.091644 * h1_2 - 2.8585005 * h1_3 - 2.258692
