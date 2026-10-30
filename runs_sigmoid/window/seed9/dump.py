# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(4.3817325 * x0 - 2.707459)  # layer 1 neuron 0
    h1_1 = sigmoid(4.5084186 * x0 - 2.7694554)  # layer 1 neuron 1
    h1_2 = sigmoid(-5.8322678 * x0 + 3.3573194)  # layer 1 neuron 2
    h1_3 = sigmoid(-8.524653 * x0 - 4.465848)  # layer 1 neuron 3
    return -1.6387875 * h1_0 - 1.6472498 * h1_1 + 3.1574929 * h1_2 - 4.954063 * h1_3 - 0.891037
