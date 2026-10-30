# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-8.082441 * x0 - 10.358925 * x1 + 4.518621)  # layer 1 neuron 0
    h1_1 = sigmoid(7.963117 * x0 - 8.720407 * x1 - 5.570991)  # layer 1 neuron 1
    h1_2 = sigmoid(5.930861 * x0 + 8.631387 * x1 - 3.5501525)  # layer 1 neuron 2
    h1_3 = sigmoid(-10.381862 * x0 - 7.5463805 * x1 + 12.785282)  # layer 1 neuron 3
    return -14.185972 * h1_0 + 5.525473 * h1_1 + 2.8010325 * h1_2 + 15.946058 * h1_3 - 10.766927
