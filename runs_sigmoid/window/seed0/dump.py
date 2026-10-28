# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(10.603771 * x0 - 5.3360934)  # layer 1 neuron 0
    h1_1 = sigmoid(12.324146 * x0 - 6.19058)  # layer 1 neuron 1
    h1_2 = sigmoid(-10.608421 * x0 - 5.24757)  # layer 1 neuron 2
    h1_3 = sigmoid(10.547431 * x0 + 5.208669)  # layer 1 neuron 3
    return -9.089059 * h1_0 - 8.819776 * h1_1 - 8.959132 * h1_2 + 8.311935 * h1_3 + 0.63069886
