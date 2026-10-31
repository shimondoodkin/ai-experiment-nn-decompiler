# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.84519124 * x0 + 0.6586871)  # layer 1 neuron 0
    h1_1 = sigmoid(-2.944553 * x0 - 2.2628136)  # layer 1 neuron 1
    h1_2 = sigmoid(0.8186878 * x0 - 0.34007758)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.30684787 * x0 - 0.18343711)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.9042168 * x0 - 5.54195)  # layer 1 neuron 4
    h1_5 = sigmoid(2.1552804 * x0 - 2.7343478)  # layer 1 neuron 5
    h1_6 = sigmoid(-2.8763556 * x0 - 5.0729666)  # layer 1 neuron 6
    h1_7 = sigmoid(3.128044 * x0 - 6.2267475)  # layer 1 neuron 7
    return -0.25824267 * h1_0 + 1.5863596 * h1_1 + 0.30481094 * h1_2 - 0.10303068 * h1_3 + 2.5045369 * h1_4 + 3.037253 * h1_5 + 2.2872417 * h1_6 + 3.1110284 * h1_7 - 0.27041385
