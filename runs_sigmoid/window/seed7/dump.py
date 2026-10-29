# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(12.153864 * x0 - 6.062294)  # layer 1 neuron 0
    h1_1 = sigmoid(-14.0477085 * x0 - 6.941848)  # layer 1 neuron 1
    h1_2 = sigmoid(-11.48867 * x0 + 5.7161884)  # layer 1 neuron 2
    h1_3 = sigmoid(11.151766 * x0 + 5.4591064)  # layer 1 neuron 3
    return -9.706537 * h1_0 - 10.269722 * h1_1 + 5.032631 * h1_2 + 4.024748 * h1_3 - 1.5382843
