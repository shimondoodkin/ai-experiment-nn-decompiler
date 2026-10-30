# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(-12.515744 * x0 - 6.27365)  # layer 1 neuron 0
    h1_1 = sigmoid(11.720995 * x0 - 5.9234037)  # layer 1 neuron 1
    h1_2 = sigmoid(12.821644 * x0 - 6.4705443)  # layer 1 neuron 2
    h1_3 = sigmoid(12.297826 * x0 + 6.151482)  # layer 1 neuron 3
    return -9.306826 * h1_0 - 7.954157 * h1_1 - 7.0211864 * h1_2 + 5.596946 * h1_3 + 1.7356448
