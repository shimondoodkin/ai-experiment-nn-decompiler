# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.30109677 * x0 + 1.0264431)  # layer 1 neuron 0
    h1_1 = tanh(-0.38681164 * x0 - 1.6788961)  # layer 1 neuron 1
    h1_2 = tanh(0.30473852 * x0 - 0.3649509)  # layer 1 neuron 2
    h1_3 = tanh(-0.067763396 * x0 + 0.5777293)  # layer 1 neuron 3
    h1_4 = tanh(0.9164171 * x0 - 0.5620068)  # layer 1 neuron 4
    h1_5 = tanh(0.011861321 * x0 + 1.8889941)  # layer 1 neuron 5
    h1_6 = tanh(-0.053528048 * x0 - 0.43595383)  # layer 1 neuron 6
    h1_7 = tanh(-2.8320155 * x0 + 1.0850652)  # layer 1 neuron 7
    return 0.45720404 * h1_0 - 0.59275985 * h1_1 - 0.18613179 * h1_2 + 0.9060557 * h1_3 - 0.7353482 * h1_4 + 0.37243947 * h1_5 - 0.1523577 * h1_6 + 1.0212694 * h1_7 + 0.37134242
