# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(-12.539128 * x0 + 6.4895496)  # layer 1 neuron 0
    h1_1 = sigmoid(3.844833 * x0 - 2.5407255)  # layer 1 neuron 1
    h1_2 = sigmoid(12.0715275 * x0 - 6.2682867)  # layer 1 neuron 2
    h1_3 = sigmoid(-12.99264 * x0 - 6.4164133)  # layer 1 neuron 3
    return 6.537737 * h1_0 - 0.7123145 * h1_1 - 7.999073 * h1_2 - 13.243555 * h1_3 + 0.31433022
