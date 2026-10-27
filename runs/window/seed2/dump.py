# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = relu(6.279147 * x0 - 2.614633)  # layer 1 neuron 0
    h1_1 = relu(-6.5160613 * x0 - 2.594116)  # layer 1 neuron 1
    h1_2 = relu(5.492867 * x0 - 2.31106)  # layer 1 neuron 2
    h1_3 = relu(-0.051077843 * x0 - 0.80846274)  # layer 1 neuron 3
    return -5.917147 * h1_0 - 8.5205145 * h1_1 - 5.8392234 * h1_2 + 0.033228934 * h1_3 + 5.6804976
