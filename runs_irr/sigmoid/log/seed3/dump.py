# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-5.429325 * x0 - 0.36790243)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.6024561 * x0 + 1.5686804)  # layer 1 neuron 1
    h1_2 = sigmoid(0.36090893 * x0 - 0.35632414)  # layer 1 neuron 2
    h1_3 = sigmoid(-2.282966 * x0 + 1.0302248)  # layer 1 neuron 3
    h1_4 = sigmoid(0.5372321 * x0 - 1.1678277)  # layer 1 neuron 4
    h1_5 = sigmoid(-6.992169 * x0 - 0.18805075)  # layer 1 neuron 5
    h1_6 = sigmoid(1.2658505 * x0 - 0.8103159)  # layer 1 neuron 6
    h1_7 = sigmoid(0.0913255 * x0 - 0.29180986)  # layer 1 neuron 7
    return -1.9416404 * h1_0 - 0.95724505 * h1_1 + 0.54732794 * h1_2 - 1.1659526 * h1_3 + 0.7470965 * h1_4 - 2.2967603 * h1_5 + 0.81691074 * h1_6 - 0.04186239 * h1_7 - 0.05936571
