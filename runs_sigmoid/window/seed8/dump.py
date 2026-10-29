# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(11.422073 * x0 - 5.739506)  # layer 1 neuron 0
    h1_1 = sigmoid(10.668881 * x0 - 5.3551435)  # layer 1 neuron 1
    h1_2 = sigmoid(11.554639 * x0 + 5.7566905)  # layer 1 neuron 2
    h1_3 = sigmoid(-11.290927 * x0 - 5.6312966)  # layer 1 neuron 3
    return -8.014564 * h1_0 - 8.179488 * h1_1 + 7.4308615 * h1_2 - 9.175122 * h1_3 + 0.693383
