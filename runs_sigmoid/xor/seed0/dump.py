# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(8.293806 * x0 + 8.294414 * x1 - 12.154446)  # layer 1 neuron 0
    h1_1 = sigmoid(-7.150811 * x0 - 7.0826163 * x1 + 2.0187764)  # layer 1 neuron 1
    h1_2 = sigmoid(-10.328212 * x0 - 10.138524 * x1 + 3.6307657)  # layer 1 neuron 2
    h1_3 = sigmoid(9.062233 * x0 + 8.9151745 * x1 - 3.0168748)  # layer 1 neuron 3
    return -17.152473 * h1_0 - 5.859491 * h1_1 - 7.6438284 * h1_2 + 7.3422647 * h1_3 + 1.6402835
