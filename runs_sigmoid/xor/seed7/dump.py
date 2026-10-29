# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-7.347241 * x0 - 7.5603952 * x1 + 2.1868026)  # layer 1 neuron 0
    h1_1 = sigmoid(8.300402 * x0 + 8.299167 * x1 - 12.198712)  # layer 1 neuron 1
    h1_2 = sigmoid(-7.784156 * x0 - 7.5621266 * x1 + 2.3117082)  # layer 1 neuron 2
    h1_3 = sigmoid(-9.43235 * x0 - 9.299206 * x1 + 3.2084236)  # layer 1 neuron 3
    return -6.555784 * h1_0 - 17.128033 * h1_1 - 5.9612546 * h1_2 - 8.57148 * h1_3 + 8.920748
