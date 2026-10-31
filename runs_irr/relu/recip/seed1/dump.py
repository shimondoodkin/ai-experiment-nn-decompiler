# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.36141175 * x0 + 0.82079893)  # layer 1 neuron 0
    h1_1 = relu(-0.44137824 * x0 - 0.12244415)  # layer 1 neuron 1
    h1_2 = relu(-0.78866607 * x0 + 0.99683607)  # layer 1 neuron 2
    h1_3 = relu(0.36317506 * x0 + 0.6282742)  # layer 1 neuron 3
    h1_4 = relu(-0.9414369 * x0 + 0.36522818)  # layer 1 neuron 4
    h1_5 = relu(0.7102314 * x0 - 0.35322058)  # layer 1 neuron 5
    h1_6 = relu(-0.20572531 * x0 - 0.07290876)  # layer 1 neuron 6
    h1_7 = relu(0.4804034 * x0 - 0.23834288)  # layer 1 neuron 7
    return 0.20133719 * h1_0 - 0.0014123917 * h1_1 + 1.6168517 * h1_2 + 0.22693245 * h1_3 - 0.13166548 * h1_4 - 0.3402343 * h1_5 - 0.059261084 * h1_6 - 0.17839943 * h1_7 + 0.4045074
