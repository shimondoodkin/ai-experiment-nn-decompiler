# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(13.332294 * x0 - 6.5483165)  # layer 1 neuron 0
    h1_1 = sigmoid(12.435798 * x0 + 6.3724947)  # layer 1 neuron 1
    h1_2 = sigmoid(-9.116508 * x0 - 4.8254514)  # layer 1 neuron 2
    h1_3 = sigmoid(-9.445804 * x0 - 4.970555)  # layer 1 neuron 3
    return -13.080802 * h1_0 + 7.2067676 * h1_1 - 3.9287426 * h1_2 - 4.455501 * h1_3 - 0.3566925
