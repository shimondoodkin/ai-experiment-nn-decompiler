# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-0.9845413 * x0 + 1.8234533)  # layer 1 neuron 0
    h1_1 = tanh(0.10121039 * x0 + 1.7378291)  # layer 1 neuron 1
    h1_2 = tanh(-0.858543 * x0 + 0.54578894)  # layer 1 neuron 2
    h1_3 = tanh(-1.0001457 * x0 + 1.9121109)  # layer 1 neuron 3
    h1_4 = tanh(-0.098426715 * x0 - 1.7730337)  # layer 1 neuron 4
    h1_5 = tanh(-0.098052435 * x0 - 1.5939656)  # layer 1 neuron 5
    h1_6 = tanh(1.5135087 * x0 - 3.197295)  # layer 1 neuron 6
    h1_7 = tanh(-0.113327414 * x0 - 1.2749594)  # layer 1 neuron 7
    return -1.2881376 * h1_0 + 2.1457565 * h1_1 - 1.0027275 * h1_2 - 1.3861502 * h1_3 - 1.1261119 * h1_4 - 1.2400333 * h1_5 + 2.3892064 * h1_6 - 1.7111098 * h1_7 + 0.738491
