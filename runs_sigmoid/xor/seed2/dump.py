# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(11.340213 * x0 - 8.89886 * x1 + 3.4267817)  # layer 1 neuron 0
    h1_1 = sigmoid(7.5984063 * x0 - 10.491862 * x1 - 2.88136)  # layer 1 neuron 1
    h1_2 = sigmoid(-6.7375093 * x0 - 6.999367 * x1 + 0.99469453)  # layer 1 neuron 2
    h1_3 = sigmoid(-6.7777934 * x0 - 7.2630124 * x1 + 1.0978357)  # layer 1 neuron 3
    return -15.786223 * h1_0 + 16.449656 * h1_1 - 3.5324583 * h1_2 - 3.6980748 * h1_3 + 8.012023
