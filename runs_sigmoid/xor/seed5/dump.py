# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(9.71923 * x0 + 6.4607253 * x1 - 11.687334)  # layer 1 neuron 0
    h1_1 = sigmoid(-10.030352 * x0 - 12.608243 * x1 + 5.200202)  # layer 1 neuron 1
    h1_2 = sigmoid(7.6301403 * x0 - 8.401334 * x1 - 5.0557766)  # layer 1 neuron 2
    h1_3 = sigmoid(-7.705192 * x0 - 9.411263 * x1 + 3.7968488)  # layer 1 neuron 3
    return -16.382105 * h1_0 - 8.22889 * h1_1 + 5.7794385 * h1_2 - 9.118613 * h1_3 + 7.8433204
