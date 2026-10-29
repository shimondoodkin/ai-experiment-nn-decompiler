# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-6.737544 * x0 + 6.7343826 * x1 + 1.5960714)  # layer 1 neuron 0
    h1_1 = sigmoid(1.6277033 * x0 + 1.2029768 * x1 + 1.9728141)  # layer 1 neuron 1
    h1_2 = sigmoid(-9.559495 * x0 + 9.548942 * x1 - 2.0907702)  # layer 1 neuron 2
    h1_3 = sigmoid(0.2775229 * x0 + 0.72566336 * x1 - 2.6518002)  # layer 1 neuron 3
    h1_4 = sigmoid(10.995403 * x0 - 10.978054 * x1 - 2.2095191)  # layer 1 neuron 4
    h1_5 = sigmoid(-10.60658 * x0 + 10.592035 * x1 - 2.2425199)  # layer 1 neuron 5
    h1_6 = sigmoid(-6.5838614 * x0 + 6.5814266 * x1 + 1.5781591)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.4170753 * x0 + 1.1491038 * x1 + 0.91076607)  # layer 1 neuron 7
    return 5.2638607 * h1_0 + 0.09734208 * h1_1 - 11.537624 * h1_2 + 1.3441408 * h1_3 - 14.602106 * h1_4 - 12.620919 * h1_5 + 4.1577406 * h1_6 + 0.9759137 * h1_7 + 1.550303
