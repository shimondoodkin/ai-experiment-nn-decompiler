# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.05662483 * x0 - 5.0086775)  # layer 1 neuron 0
    h1_1 = sigmoid(0.08329499 * x0 - 11.832382)  # layer 1 neuron 1
    h1_2 = sigmoid(0.08305155 * x0 - 14.739377)  # layer 1 neuron 2
    h1_3 = sigmoid(0.072534 * x0 - 9.11031)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.08553783 * x0 + 8.183611)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.09257642 * x0 + 8.716337)  # layer 1 neuron 5
    h1_6 = sigmoid(0.085790575 * x0 - 15.323278)  # layer 1 neuron 6
    h1_7 = sigmoid(0.08098933 * x0 - 2.7229297)  # layer 1 neuron 7
    return 9.2457905 * h1_0 + 9.074245 * h1_1 + 8.965652 * h1_2 + 8.84185 * h1_3 - 4.8894677 * h1_4 - 4.5634494 * h1_5 + 8.991322 * h1_6 + 9.23601 * h1_7 + 9.064263
