# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(-13.056992 * x0 - 6.5731335)  # layer 1 neuron 0
    h1_1 = sigmoid(-11.319427 * x0 - 5.7056246)  # layer 1 neuron 1
    h1_2 = sigmoid(9.77894 * x0 - 4.816538)  # layer 1 neuron 2
    h1_3 = sigmoid(-13.89691 * x0 + 6.89883)  # layer 1 neuron 3
    return -7.9286985 * h1_0 - 7.8505735 * h1_1 - 5.6845927 * h1_2 + 9.229124 * h1_3 - 1.75977
