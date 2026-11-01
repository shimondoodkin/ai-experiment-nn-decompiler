# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(1.9214575 * x0 + 0.4519634)  # layer 1 neuron 0
    h1_1 = sigmoid(0.48939317 * x0 - 0.7263431)  # layer 1 neuron 1
    h1_2 = sigmoid(0.59655595 * x0 - 0.1181817)  # layer 1 neuron 2
    h1_3 = sigmoid(0.620857 * x0 - 0.5089339)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.9720135 * x0 - 4.590729)  # layer 1 neuron 4
    h1_5 = sigmoid(0.50817275 * x0 - 1.0235864)  # layer 1 neuron 5
    h1_6 = sigmoid(0.5059482 * x0 - 1.2974042)  # layer 1 neuron 6
    h1_7 = sigmoid(0.5091234 * x0 - 1.2464134)  # layer 1 neuron 7
    return 0.48332086 * h1_0 + 0.17951503 * h1_1 + 0.5280978 * h1_2 + 0.4910955 * h1_3 - 1.739697 * h1_4 + 0.21284248 * h1_5 + 0.24033755 * h1_6 + 0.26914033 * h1_7 - 0.2471542
