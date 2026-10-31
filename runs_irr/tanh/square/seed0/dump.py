# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.95803124 * x0 - 1.142343)  # layer 1 neuron 0
    h1_1 = tanh(-0.016433913 * x0 + 1.5018904)  # layer 1 neuron 1
    h1_2 = tanh(0.020744633 * x0 - 1.6749123)  # layer 1 neuron 2
    h1_3 = tanh(-1.426893 * x0 - 2.8349195)  # layer 1 neuron 3
    h1_4 = tanh(0.009344748 * x0 - 1.7458495)  # layer 1 neuron 4
    h1_5 = tanh(1.5006529 * x0 - 3.0457249)  # layer 1 neuron 5
    h1_6 = tanh(0.023045348 * x0 - 1.4839638)  # layer 1 neuron 6
    h1_7 = tanh(1.0753973 * x0 + 1.0783869)  # layer 1 neuron 7
    return 1.7710521 * h1_0 + 2.1819944 * h1_1 - 1.9257618 * h1_2 + 2.159261 * h1_3 - 0.9299204 * h1_4 + 1.78606 * h1_5 - 1.1553223 * h1_6 - 1.3177993 * h1_7 + 0.7232189
