# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.08143656 * x0 + 9.816497)  # layer 1 neuron 0
    h1_1 = sigmoid(0.020150246 * x0 + 6.02202)  # layer 1 neuron 1
    h1_2 = sigmoid(0.06303804 * x0 - 2.0999393)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.08421991 * x0 + 14.67481)  # layer 1 neuron 3
    h1_4 = sigmoid(0.049849574 * x0 - 4.740201)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.09408093 * x0 + 8.733573)  # layer 1 neuron 5
    h1_6 = sigmoid(0.08799416 * x0 - 15.660592)  # layer 1 neuron 6
    h1_7 = sigmoid(0.08064836 * x0 - 11.411838)  # layer 1 neuron 7
    return -8.028272 * h1_0 + 12.382518 * h1_1 + 11.064467 * h1_2 - 7.7539163 * h1_3 + 10.1488495 * h1_4 - 7.483179 * h1_5 + 10.206247 * h1_6 + 9.494898 * h1_7 + 9.595372
