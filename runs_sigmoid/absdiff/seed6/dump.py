# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.0794313 * x0 - 0.22849768 * x1 + 1.5651078)  # layer 1 neuron 0
    h1_1 = sigmoid(0.24235995 * x0 + 0.099429205 * x1 + 1.4266796)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.0500133 * x0 + 0.91374284 * x1 - 3.231169)  # layer 1 neuron 2
    h1_3 = sigmoid(-2.5752344 * x0 + 2.5738504 * x1 - 2.190876)  # layer 1 neuron 3
    h1_4 = sigmoid(1.1484524 * x0 - 1.153442 * x1 - 4.1819715)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.8182395 * x0 + 0.95643985 * x1 - 3.4904284)  # layer 1 neuron 5
    h1_6 = sigmoid(0.1800595 * x0 + 0.0008439808 * x1 + 1.1421691)  # layer 1 neuron 6
    h1_7 = sigmoid(-2.2319555 * x0 + 2.2395923 * x1 + 2.075663)  # layer 1 neuron 7
    return 0.59687936 * h1_0 + 0.32079685 * h1_1 + 2.4312408 * h1_2 + 1.5796996 * h1_3 + 3.9647257 * h1_4 + 2.5492377 * h1_5 + 0.38356742 * h1_6 - 2.0123472 * h1_7 + 0.58172214
