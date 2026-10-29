# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = tanh(-0.013197124 * x0 - 1.3732675)  # layer 1 neuron 0
    h1_1 = tanh(0.015426634 * x0 + 0.9846172)  # layer 1 neuron 1
    h1_2 = tanh(-0.019144239 * x0 - 0.48475167)  # layer 1 neuron 2
    h1_3 = tanh(0.01914296 * x0 + 0.50413185)  # layer 1 neuron 3
    return 0.44085896 * h1_0 - 0.39398763 * h1_1 + 0.33876163 * h1_2 - 0.1684313 * h1_3 - 0.2082389
