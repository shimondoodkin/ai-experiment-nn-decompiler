# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(6.955094 * x0 + 6.880714 * x1 - 10.368961)  # layer 1 neuron 0
    h1_1 = sigmoid(-8.500082 * x0 - 8.175231 * x1 + 3.685046)  # layer 1 neuron 1
    h1_2 = sigmoid(6.0475545 * x0 + 6.0804734 * x1 - 9.073109)  # layer 1 neuron 2
    h1_3 = sigmoid(8.332502 * x0 + 8.420844 * x1 - 3.6962357)  # layer 1 neuron 3
    return -10.722438 * h1_0 - 10.440861 * h1_1 - 9.782844 * h1_2 + 10.067116 * h1_3 + 0.3506789
