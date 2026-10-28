# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.08681871 * x0 - 8.035956)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.102661215 * x0 + 3.9769635)  # layer 1 neuron 1
    h1_2 = sigmoid(0.034065925 * x0 + 0.45032874)  # layer 1 neuron 2
    h1_3 = sigmoid(0.073165216 * x0 - 9.235842)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.08977439 * x0 + 12.048067)  # layer 1 neuron 4
    h1_5 = sigmoid(0.090636685 * x0 - 16.341398)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.064846985 * x0 + 6.028117)  # layer 1 neuron 6
    h1_7 = sigmoid(0.08884129 * x0 - 14.700708)  # layer 1 neuron 7
    return 10.39346 * h1_0 - 6.054742 * h1_1 + 11.280421 * h1_2 + 10.4607525 * h1_3 - 5.374301 * h1_4 + 10.127714 * h1_5 - 5.8925333 * h1_6 + 10.167235 * h1_7 + 10.210651
