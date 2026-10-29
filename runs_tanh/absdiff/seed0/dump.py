# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = tanh(0.02685782 * x0 + 0.029734058 * x1 + 1.4272918)  # layer 1 neuron 0
    h1_1 = tanh(-0.03329181 * x0 - 0.029165335 * x1 + 1.3440735)  # layer 1 neuron 1
    h1_2 = tanh(-0.38088593 * x0 + 0.38900813 * x1 - 0.7866674)  # layer 1 neuron 2
    h1_3 = tanh(0.02962898 * x0 + 0.025462113 * x1 - 1.1097628)  # layer 1 neuron 3
    h1_4 = tanh(-0.3784108 * x0 + 0.3762217 * x1 + 0.7849715)  # layer 1 neuron 4
    h1_5 = tanh(-0.085820794 * x0 - 0.080495246 * x1 + 1.2818668)  # layer 1 neuron 5
    h1_6 = tanh(-0.05922657 * x0 - 0.06235523 * x1 - 1.179106)  # layer 1 neuron 6
    h1_7 = tanh(0.022196637 * x0 + 0.026969722 * x1 + 1.1889496)  # layer 1 neuron 7
    return 1.0768071 * h1_0 + 0.92706853 * h1_1 + 3.799525 * h1_2 - 1.2873737 * h1_3 - 3.871807 * h1_4 + 0.5983937 * h1_5 - 1.1552155 * h1_6 + 1.0367786 * h1_7 + 0.35808393
