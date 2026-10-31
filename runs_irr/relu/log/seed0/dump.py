# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.088744044)  # layer 1 neuron 0
    h1_1 = relu(0.35112748 * x0 - 0.022285163)  # layer 1 neuron 1
    h1_2 = relu(-0.82304513 * x0 - 0.30221307)  # layer 1 neuron 2
    h1_3 = relu(-0.735939 * x0 - 0.19656539)  # layer 1 neuron 3
    h1_4 = relu(-0.38515437 * x0 - 0.9553485)  # layer 1 neuron 4
    h1_5 = relu(0.15446143 * x0 - 0.0034372762)  # layer 1 neuron 5
    h1_6 = relu(-0.01981318 * x0 - 0.4122231)  # layer 1 neuron 6
    h1_7 = relu(0.96963644 * x0 - 0.0999562)  # layer 1 neuron 7
    return 0.13977209 * h1_0 + 0.14429575 * h1_1 - 0.23968837 * h1_2 - 0.15395936 * h1_3 + 0.12841663 * h1_4 + 0.054546803 * h1_5 - 0.07276136 * h1_6 + 0.49121305 * h1_7 - 0.655454
