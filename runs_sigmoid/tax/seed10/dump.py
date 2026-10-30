# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.077428326 * x0 + 9.821615)  # layer 1 neuron 0
    h1_1 = sigmoid(0.05532139 * x0 - 2.5241287)  # layer 1 neuron 1
    h1_2 = sigmoid(0.02366253 * x0 + 0.2693086)  # layer 1 neuron 2
    h1_3 = sigmoid(0.08395844 * x0 - 14.317049)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.09108975 * x0 + 12.115653)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.06970989 * x0 + 8.2675905)  # layer 1 neuron 5
    h1_6 = sigmoid(0.082829736 * x0 - 14.818984)  # layer 1 neuron 6
    h1_7 = sigmoid(0.10365881 * x0 - 9.775188)  # layer 1 neuron 7
    return -5.425409 * h1_0 + 10.7551155 * h1_1 + 11.494866 * h1_2 + 10.350927 * h1_3 - 6.6094913 * h1_4 - 5.4802823 * h1_5 + 10.056583 * h1_6 + 10.058161 * h1_7 + 9.972204
