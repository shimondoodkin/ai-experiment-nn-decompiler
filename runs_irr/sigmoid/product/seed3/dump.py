# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.80303454 * x0 - 0.8144918 * x1 - 3.0338042)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.81879187 * x0 - 0.77180225 * x1 - 2.8679137)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.9483143 * x0 - 0.9430437 * x1 - 3.872277)  # layer 1 neuron 2
    h1_3 = sigmoid(1.5292504 * x0 - 1.5771661 * x1 - 6.023418)  # layer 1 neuron 3
    h1_4 = sigmoid(-1.5474333 * x0 + 1.6353706 * x1 + 2.514622)  # layer 1 neuron 4
    h1_5 = sigmoid(1.4548604 * x0 + 1.4838537 * x1 - 5.487598)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.7867307 * x0 - 0.79321253 * x1 - 2.8107355)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.5660866 * x0 + 1.6385514 * x1 - 2.0545492)  # layer 1 neuron 7
    h1_8 = sigmoid(-0.86744064 * x0 + 0.97544265 * x1 - 3.319585)  # layer 1 neuron 8
    h1_9 = sigmoid(1.0029857 * x0 + 0.98852557 * x1 - 3.2973154)  # layer 1 neuron 9
    h1_10 = sigmoid(1.1256912 * x0 - 1.0583256 * x1 + 4.2144265)  # layer 1 neuron 10
    h1_11 = sigmoid(1.1631877 * x0 - 1.0885935 * x1 - 3.5996065)  # layer 1 neuron 11
    h1_12 = sigmoid(0.9784579 * x0 - 0.5608147 * x1 - 0.97271913)  # layer 1 neuron 12
    h1_13 = sigmoid(-1.1809945 * x0 + 1.0748719 * x1 - 4.3787475)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.3026369 * x0 - 1.3073443 * x1 + 1.9017779)  # layer 1 neuron 14
    h1_15 = sigmoid(-0.8594794 * x0 + 0.96841717 * x1 - 3.261018)  # layer 1 neuron 15
    return 1.8144825 * h1_0 + 2.148099 * h1_1 + 2.132813 * h1_2 - 2.4725153 * h1_3 + 0.640372 * h1_4 + 3.0219076 * h1_5 + 1.96424 * h1_6 - 0.5528745 * h1_7 - 1.8557255 * h1_8 + 2.376381 * h1_9 + 0.8692871 * h1_10 - 2.6641467 * h1_11 - 0.6568404 * h1_12 - 2.0567062 * h1_13 - 1.3554099 * h1_14 - 2.1142054 * h1_15 - 0.23003462
