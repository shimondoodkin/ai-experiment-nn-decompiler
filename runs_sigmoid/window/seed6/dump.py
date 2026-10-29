# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(-13.635628 * x0 - 6.8473167)  # layer 1 neuron 0
    h1_1 = sigmoid(10.848226 * x0 - 5.9092326)  # layer 1 neuron 1
    h1_2 = sigmoid(10.799769 * x0 - 5.891862)  # layer 1 neuron 2
    h1_3 = sigmoid(12.682284 * x0 - 6.7334433)  # layer 1 neuron 3
    return -12.595198 * h1_0 - 6.336562 * h1_1 - 5.350642 * h1_2 - 4.2973676 * h1_3 + 6.1496286
