# dumped from a neural network: 2 inputs -> 8 -> 3 outputs
# inputs: x0: float [1, 6.9], x1: float [0.1, 2.5]
# output: argmax over 3 outputs: the returned class index
def net(x0, x1):
    h1_0 = sigmoid(-0.78486806 * x0 - 0.78341347 * x1 + 4.7726398)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.0510789 * x0 - 3.7224672 * x1 + 5.0972114)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.1883245 * x0 - 3.433992 * x1 + 5.219406)  # layer 1 neuron 2
    h1_3 = sigmoid(0.5533285 * x0 + 5.1377635 * x1 - 4.9987936)  # layer 1 neuron 3
    h1_4 = sigmoid(0.6937207 * x0 + 0.87599885 * x1 - 4.0766973)  # layer 1 neuron 4
    h1_5 = sigmoid(0.81895435 * x0 + 1.0743415 * x1 - 5.688247)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.8815541 * x0 - 0.7268376 * x1 + 5.4194217)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.66054714 * x0 - 0.90852493 * x1 + 4.07653)  # layer 1 neuron 7
    o0 = 2.306797 * h1_0 + 2.143836 * h1_1 + 2.362905 * h1_2 - 4.5157423 * h1_3 - 3.0926309 * h1_4 - 2.674745 * h1_5 + 1.2867624 * h1_6 + 2.3799043 * h1_7 - 0.9827956  # output 0
    o1 = 2.8226519 * h1_0 - 4.650317 * h1_1 - 5.377497 * h1_2 + 0.95900106 * h1_3 - 1.6647309 * h1_4 - 4.000581 * h1_5 + 3.1017518 * h1_6 + 2.9140646 * h1_7 + 0.18886518  # output 1
    o2 = -4.951103 * h1_0 - 1.2532047 * h1_1 - 1.242583 * h1_2 + 0.8980451 * h1_3 + 3.7825124 * h1_4 + 4.340224 * h1_5 - 3.802498 * h1_6 - 4.13405 * h1_7 + 0.83119714  # output 2
    return argmax(o0, o1, o2)
