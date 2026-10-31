# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(1.6738 * x0 - 3.4861035)  # layer 1 neuron 0
    h1_1 = tanh(0.056967776 * x0 + 1.3194762)  # layer 1 neuron 1
    h1_2 = tanh(-0.056428965 * x0 - 1.4122105)  # layer 1 neuron 2
    h1_3 = tanh(-0.056388125 * x0 - 1.3465078)  # layer 1 neuron 3
    h1_4 = tanh(-0.052141525 * x0 - 1.8470768)  # layer 1 neuron 4
    h1_5 = tanh(0.6390307 * x0 - 0.6243669)  # layer 1 neuron 5
    h1_6 = tanh(-0.05538155 * x0 - 1.4490687)  # layer 1 neuron 6
    h1_7 = tanh(0.9846969 * x0 - 1.4312159)  # layer 1 neuron 7
    return 2.9354713 * h1_0 + 1.3394998 * h1_1 - 1.3547778 * h1_2 - 1.2869484 * h1_3 - 0.79093444 * h1_4 + 1.3542061 * h1_5 - 1.199706 * h1_6 + 1.5697277 * h1_7 + 0.76193935
