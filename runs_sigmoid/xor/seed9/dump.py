# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(6.817937 * x0 + 6.401187 * x1 - 10.302943)  # layer 1 neuron 0
    h1_1 = sigmoid(-8.005645 * x0 + 7.7703333 * x1 + 4.4709954)  # layer 1 neuron 1
    h1_2 = sigmoid(8.230583 * x0 + 8.540629 * x1 - 4.004668)  # layer 1 neuron 2
    h1_3 = sigmoid(-8.145661 * x0 + 8.625192 * x1 - 4.6189775)  # layer 1 neuron 3
    return -10.795106 * h1_0 - 10.278005 * h1_1 + 9.37419 * h1_2 + 9.556478 * h1_3 + 0.93858474
