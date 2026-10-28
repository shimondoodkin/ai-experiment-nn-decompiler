# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.5760342 * x0 + 1.5582697 * x1 + 1.9079688)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.918844 * x0 - 1.9395111 * x1 + 1.2231603)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.1844841 * x0 + 1.1826217 * x1 - 4.360774)  # layer 1 neuron 2
    h1_3 = sigmoid(-2.335223 * x0 + 2.3351378 * x1 - 2.3598416)  # layer 1 neuron 3
    h1_4 = sigmoid(2.1175911 * x0 - 2.1213229 * x1 - 2.2691188)  # layer 1 neuron 4
    h1_5 = sigmoid(-1.670568 * x0 - 1.6375492 * x1 + 1.6302974)  # layer 1 neuron 5
    h1_6 = sigmoid(1.2869285 * x0 - 1.2854285 * x1 - 4.8206944)  # layer 1 neuron 6
    h1_7 = sigmoid(1.386017 * x0 + 1.3439522 * x1 + 1.0225431)  # layer 1 neuron 7
    return -0.19723664 * h1_0 + 0.14666697 * h1_1 + 3.9260743 * h1_2 + 1.9408692 * h1_3 + 2.1633446 * h1_4 - 0.0724625 * h1_5 + 3.6088986 * h1_6 + 0.26983872 * h1_7 - 0.27117229
