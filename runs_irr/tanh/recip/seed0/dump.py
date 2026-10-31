# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-2.1660783 * x0 + 0.4451902)  # layer 1 neuron 0
    h1_1 = tanh(0.80721474 * x0 + 1.1050626)  # layer 1 neuron 1
    h1_2 = tanh(-1.343138 * x0 - 1.376155)  # layer 1 neuron 2
    h1_3 = tanh(-1.1412438 * x0 - 1.2624274)  # layer 1 neuron 3
    h1_4 = tanh(0.9204556 * x0 - 0.90604705)  # layer 1 neuron 4
    h1_5 = tanh(0.22562441 * x0 - 0.82744366)  # layer 1 neuron 5
    h1_6 = tanh(2.0055382 * x0 - 0.37161332)  # layer 1 neuron 6
    h1_7 = tanh(1.1316706 * x0 + 0.79798275)  # layer 1 neuron 7
    return 1.25352 * h1_0 + 0.55947673 * h1_1 - 0.82623404 * h1_2 - 0.56687695 * h1_3 - 0.46834624 * h1_4 - 0.31231895 * h1_5 - 1.0429138 * h1_6 + 0.37482667 * h1_7 + 0.70250964
