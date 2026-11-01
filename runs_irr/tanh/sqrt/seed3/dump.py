# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-1.7802414 * x0 - 0.6105136)  # layer 1 neuron 0
    h1_1 = tanh(-0.33924994 * x0 + 1.0901135)  # layer 1 neuron 1
    h1_2 = tanh(-0.21270971 * x0 + 0.17497751)  # layer 1 neuron 2
    h1_3 = tanh(-0.9705576 * x0 + 0.8939419)  # layer 1 neuron 3
    h1_4 = tanh(-0.22749662 * x0 - 0.556148)  # layer 1 neuron 4
    h1_5 = tanh(-1.9308541 * x0 - 0.64263266)  # layer 1 neuron 5
    h1_6 = tanh(0.30236262 * x0 - 0.74513704)  # layer 1 neuron 6
    h1_7 = tanh(0.14866842 * x0 - 0.3646647)  # layer 1 neuron 7
    return -0.5748597 * h1_0 - 0.11325959 * h1_1 - 0.11380507 * h1_2 - 0.18411124 * h1_3 - 0.21677224 * h1_4 - 0.55107254 * h1_5 + 0.69297254 * h1_6 + 0.015543765 * h1_7 + 0.09290676
