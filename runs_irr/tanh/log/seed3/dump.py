# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-4.145116 * x0 + 0.21357483)  # layer 1 neuron 0
    h1_1 = tanh(-0.9593861 * x0 + 1.3448228)  # layer 1 neuron 1
    h1_2 = tanh(-0.13955052 * x0 + 0.3236823)  # layer 1 neuron 2
    h1_3 = tanh(-2.02856 * x0 + 1.304789)  # layer 1 neuron 3
    h1_4 = tanh(-0.08863698 * x0 - 1.4846894)  # layer 1 neuron 4
    h1_5 = tanh(-3.0910537 * x0 + 0.12651247)  # layer 1 neuron 5
    h1_6 = tanh(0.3291047 * x0 - 1.0047626)  # layer 1 neuron 6
    h1_7 = tanh(0.18409926 * x0 + 0.6575354)  # layer 1 neuron 7
    return -0.9234505 * h1_0 - 0.37934366 * h1_1 - 0.036974274 * h1_2 - 0.41502392 * h1_3 + 0.55410135 * h1_4 - 0.7212033 * h1_5 + 0.8461507 * h1_6 - 0.21963045 * h1_7 - 0.5903307
