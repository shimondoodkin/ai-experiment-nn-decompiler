# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-8.8685055 * x0 - 5.927287 * x1 + 10.538361)  # layer 1 neuron 0
    h1_1 = sigmoid(-9.445049 * x0 - 6.5768886 * x1 + 11.477547)  # layer 1 neuron 1
    h1_2 = sigmoid(-8.801477 * x0 - 10.839543 * x1 + 5.016882)  # layer 1 neuron 2
    h1_3 = sigmoid(8.641885 * x0 - 8.952727 * x1 - 5.726291)  # layer 1 neuron 3
    return 8.82249 * h1_0 + 7.7790647 * h1_1 - 16.466808 * h1_2 + 5.449029 * h1_3 - 8.447832
