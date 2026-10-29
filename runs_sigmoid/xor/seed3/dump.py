# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-6.589475 * x0 - 12.774582 * x1 + 1.460924)  # layer 1 neuron 0
    h1_1 = sigmoid(8.259559 * x0 - 15.281261 * x1 - 4.893095)  # layer 1 neuron 1
    h1_2 = sigmoid(-6.621455 * x0 - 13.72509 * x1 + 1.9255176)  # layer 1 neuron 2
    h1_3 = sigmoid(5.74104 * x0 + 11.729524 * x1 - 0.7797679)  # layer 1 neuron 3
    return -5.329522 * h1_0 + 9.517665 * h1_1 - 5.525938 * h1_2 + 1.3433632 * h1_3 - 1.3431742
