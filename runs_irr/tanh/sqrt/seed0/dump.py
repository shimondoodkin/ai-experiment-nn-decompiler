# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.14812261 * x0 - 0.056964155)  # layer 1 neuron 0
    h1_1 = tanh(0.61671627 * x0 + 0.022982689)  # layer 1 neuron 1
    h1_2 = tanh(-3.7671535 * x0 - 0.5399646)  # layer 1 neuron 2
    h1_3 = tanh(-0.6341597 * x0 - 0.40400305)  # layer 1 neuron 3
    h1_4 = tanh(-0.16777511 * x0 - 0.7888052)  # layer 1 neuron 4
    h1_5 = tanh(0.25631788 * x0 - 0.6903897)  # layer 1 neuron 5
    h1_6 = tanh(-0.17304286 * x0 - 0.39874294)  # layer 1 neuron 6
    h1_7 = tanh(0.969187 * x0 + 0.6306423)  # layer 1 neuron 7
    return 0.21529223 * h1_0 + 0.3469439 * h1_1 - 0.49294677 * h1_2 - 0.25546247 * h1_3 + 0.013290155 * h1_4 + 0.8637539 * h1_5 - 0.1898984 * h1_6 + 0.34818643 * h1_7 + 0.034834675
