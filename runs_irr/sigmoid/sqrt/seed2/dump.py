# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.61795855 * x0 - 0.6259632)  # layer 1 neuron 0
    h1_1 = sigmoid(-3.1447363 * x0 - 0.94157207)  # layer 1 neuron 1
    h1_2 = sigmoid(0.50364226 * x0 - 0.7963141)  # layer 1 neuron 2
    h1_3 = sigmoid(0.56103325 * x0 - 0.33833876)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.19035846 * x0 - 0.3595415)  # layer 1 neuron 4
    h1_5 = sigmoid(3.055076 * x0 + 0.7740911)  # layer 1 neuron 5
    h1_6 = sigmoid(0.7415675 * x0 - 1.163524)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.43476355 * x0 + 0.7358729)  # layer 1 neuron 7
    return 0.55879945 * h1_0 - 1.384384 * h1_1 + 0.25238144 * h1_2 + 0.7362891 * h1_3 - 0.1978646 * h1_4 + 0.3987375 * h1_5 + 0.5036456 * h1_6 - 0.55852735 * h1_7 + 0.03712076
