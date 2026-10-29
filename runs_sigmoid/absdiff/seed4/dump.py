# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.052513428 * x0 - 0.22641179 * x1 + 1.3441563)  # layer 1 neuron 0
    h1_1 = sigmoid(0.09996305 * x0 - 0.07863644 * x1 + 1.2677885)  # layer 1 neuron 1
    h1_2 = sigmoid(0.04891198 * x0 - 0.17902395 * x1 + 0.9946669)  # layer 1 neuron 2
    h1_3 = sigmoid(0.710797 * x0 - 0.7152358 * x1 - 1.7827095)  # layer 1 neuron 3
    h1_4 = sigmoid(0.19357155 * x0 - 0.005729861 * x1 + 1.1464425)  # layer 1 neuron 4
    h1_5 = sigmoid(1.2538643 * x0 - 1.253644 * x1 + 4.500368)  # layer 1 neuron 5
    h1_6 = sigmoid(0.29027292 * x0 - 0.15421104 * x1 + 0.52731156)  # layer 1 neuron 6
    h1_7 = sigmoid(-2.5734644 * x0 + 2.5779605 * x1 - 2.6494086)  # layer 1 neuron 7
    return 1.0139246 * h1_0 + 0.8933184 * h1_1 + 0.9775807 * h1_2 + 5.9141073 * h1_3 + 1.0562305 * h1_4 - 4.4847927 * h1_5 + 0.9540187 * h1_6 + 2.4610639 * h1_7 + 0.12999307
