# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(1.7701871 * x0 - 0.36283207)  # layer 1 neuron 0
    h1_1 = tanh(-0.19600782 * x0 + 1.0346931)  # layer 1 neuron 1
    h1_2 = tanh(0.024421034 * x0 + 0.7487953)  # layer 1 neuron 2
    h1_3 = tanh(1.0911983 * x0 - 0.58180994)  # layer 1 neuron 3
    h1_4 = tanh(-4.5252895 * x0 + 0.20119184)  # layer 1 neuron 4
    h1_5 = tanh(0.72024345 * x0 - 1.1741852)  # layer 1 neuron 5
    h1_6 = tanh(-0.32300252 * x0 + 0.9031833)  # layer 1 neuron 6
    h1_7 = tanh(-0.009197115 * x0 - 1.7059321)  # layer 1 neuron 7
    return 0.5441598 * h1_0 - 0.4473996 * h1_1 - 0.51694256 * h1_2 + 0.561134 * h1_3 - 1.1935728 * h1_4 + 0.27804658 * h1_5 - 0.53762054 * h1_6 + 0.3976388 * h1_7 - 0.5275753
