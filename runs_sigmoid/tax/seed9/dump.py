# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.08141469 * x0 - 11.557726)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.065679714 * x0 + 6.4288836)  # layer 1 neuron 1
    h1_2 = sigmoid(0.0749188 * x0 - 2.588092)  # layer 1 neuron 2
    h1_3 = sigmoid(0.090826795 * x0 - 8.577031)  # layer 1 neuron 3
    h1_4 = sigmoid(0.08440475 * x0 - 15.142142)  # layer 1 neuron 4
    h1_5 = sigmoid(0.08171116 * x0 - 14.524111)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.047793485 * x0 + 3.982503)  # layer 1 neuron 6
    h1_7 = sigmoid(0.07340462 * x0 - 9.29138)  # layer 1 neuron 7
    return 8.859842 * h1_0 - 4.7449703 * h1_1 + 9.740518 * h1_2 + 8.941146 * h1_3 + 9.056866 * h1_4 + 9.131381 * h1_5 - 4.629024 * h1_6 + 9.011615 * h1_7 + 8.782484
