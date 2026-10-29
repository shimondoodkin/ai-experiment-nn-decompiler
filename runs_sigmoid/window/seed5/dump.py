# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(-9.792134 * x0 - 4.807528)  # layer 1 neuron 0
    h1_1 = sigmoid(10.053341 * x0 - 5.0705786)  # layer 1 neuron 1
    h1_2 = sigmoid(11.440424 * x0 - 5.7717667)  # layer 1 neuron 2
    h1_3 = sigmoid(11.187054 * x0 + 5.5143566)  # layer 1 neuron 3
    return -8.544948 * h1_0 - 8.598668 * h1_1 - 9.444186 * h1_2 + 9.055646 * h1_3 - 0.15467685
