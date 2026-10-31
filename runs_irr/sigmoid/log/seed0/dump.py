# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.47670236 * x0 - 0.9982923)  # layer 1 neuron 0
    h1_1 = sigmoid(1.3817441 * x0 - 0.63582003)  # layer 1 neuron 1
    h1_2 = sigmoid(-7.1801314 * x0 - 0.19867592)  # layer 1 neuron 2
    h1_3 = sigmoid(-7.6110067 * x0 - 0.13661525)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.0871327 * x0 + 0.72847813)  # layer 1 neuron 4
    h1_5 = sigmoid(0.52160037 * x0 - 1.600172)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.67327917 * x0 + 0.9945006)  # layer 1 neuron 6
    h1_7 = sigmoid(3.1382196 * x0 - 0.049957708)  # layer 1 neuron 7
    return 0.41093612 * h1_0 + 0.5856719 * h1_1 - 1.7518322 * h1_2 - 1.6952989 * h1_3 - 1.2430457 * h1_4 + 0.87523925 * h1_5 - 1.0958809 * h1_6 + 0.69860524 * h1_7 - 0.55407816
