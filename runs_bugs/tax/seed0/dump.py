# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.088744044)  # layer 1 neuron 0
    h1_1 = relu(0.2008872 * x0 - 4.5912776)  # layer 1 neuron 1
    h1_2 = relu(-0.82304513 * x0 - 0.30221307)  # layer 1 neuron 2
    h1_3 = relu(-0.735939 * x0 - 0.19656539)  # layer 1 neuron 3
    h1_4 = relu(-0.38515437 * x0 - 0.9553485)  # layer 1 neuron 4
    h1_5 = relu(0.15241578 * x0 - 9.662828)  # layer 1 neuron 5
    h1_6 = relu(-0.01981318 * x0 - 0.4122231)  # layer 1 neuron 6
    h1_7 = relu(0.6122562 * x0 - 2.0766058)  # layer 1 neuron 7
    return 0.13977209 * h1_0 + 0.1978575 * h1_1 - 0.23968837 * h1_2 - 0.15395936 * h1_3 + 0.12841663 * h1_4 + 1.2616787 * h1_5 - 0.07276136 * h1_6 + 0.24540089 * h1_7 - 0.154781
