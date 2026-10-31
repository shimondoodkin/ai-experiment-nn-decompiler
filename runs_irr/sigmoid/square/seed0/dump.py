# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(2.8280725 * x0 - 2.3221865)  # layer 1 neuron 0
    h1_1 = sigmoid(0.5209724 * x0 - 0.34604058)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.5119346 * x0 + 1.4552888)  # layer 1 neuron 2
    h1_3 = sigmoid(-2.4178953 * x0 - 2.171306)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.2916186 * x0 - 4.4098334)  # layer 1 neuron 4
    h1_5 = sigmoid(2.7040038 * x0 - 5.3459783)  # layer 1 neuron 5
    h1_6 = sigmoid(-2.7891488 * x0 - 5.606526)  # layer 1 neuron 6
    h1_7 = sigmoid(2.008472 * x0 - 3.3436847)  # layer 1 neuron 7
    return 0.9455018 * h1_0 + 0.17974727 * h1_1 - 0.38413697 * h1_2 + 1.7719474 * h1_3 + 3.0152574 * h1_4 + 3.7551546 * h1_5 + 2.390291 * h1_6 + 1.84284 * h1_7 - 0.15337221
