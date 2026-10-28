# dumped from a neural network: 2 inputs -> 8 -> 3 outputs
# inputs: x0: float [1, 6.9], x1: float [0.1, 2.5]
# output: argmax over 3 outputs: the returned class index
def net(x0, x1):
    h1_0 = sigmoid(0.75291437 * x0 + 0.9981506 * x1 - 4.7510695)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.86998254 * x0 - 3.02555 * x1 + 3.8740008)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.1588602 * x0 - 2.673463 * x1 + 4.3564034)  # layer 1 neuron 2
    h1_3 = sigmoid(0.8132326 * x0 + 0.8650321 * x1 - 4.117611)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.876331 * x0 - 0.7369177 * x1 + 5.4177823)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.8293763 * x0 - 1.0147086 * x1 + 5.678925)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.7421966 * x0 - 0.96439767 * x1 + 4.7099404)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.3849044 * x0 - 2.5346978 * x1 + 4.9175243)  # layer 1 neuron 7
    o0 = -3.2452078 * h1_0 + 2.2151215 * h1_1 + 3.4629858 * h1_2 - 3.2752929 * h1_3 + 1.3324376 * h1_4 + 1.1294788 * h1_5 + 1.76383 * h1_6 + 2.7689366 * h1_7 - 2.8476467  # output 0
    o1 = -2.483379 * h1_0 - 3.1418302 * h1_1 - 5.637339 * h1_2 - 0.7711441 * h1_3 + 4.0681543 * h1_4 + 4.5862308 * h1_5 + 2.571128 * h1_6 - 4.1931987 * h1_7 - 0.5202746  # output 1
    o2 = 4.425873 * h1_0 - 1.8535136 * h1_1 - 1.6078502 * h1_2 + 3.2966425 * h1_3 - 4.4366326 * h1_4 - 3.9088402 * h1_5 - 4.450044 * h1_6 - 1.1066092 * h1_7 + 1.8857558  # output 2
    return argmax(o0, o1, o2)
