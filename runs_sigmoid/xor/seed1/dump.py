# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-7.606464 * x0 + 7.702796 * x1 - 5.1910515)  # layer 1 neuron 0
    h1_1 = sigmoid(-8.549309 * x0 + 8.592343 * x1 - 5.5618186)  # layer 1 neuron 1
    h1_2 = sigmoid(-9.960342 * x0 + 10.273031 * x1 + 4.888616)  # layer 1 neuron 2
    h1_3 = sigmoid(-6.7263336 * x0 + 6.6583986 * x1 - 4.750122)  # layer 1 neuron 3
    return 7.441428 * h1_0 + 8.162767 * h1_1 - 16.808912 * h1_2 + 5.468178 * h1_3 + 7.9810863
