# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-1.684484 * x0 + 2.0843925)  # layer 1 neuron 0
    h1_1 = relu(-1.6220877 * x0 + 0.91993684)  # layer 1 neuron 1
    h1_2 = relu(-1.1702704 * x0 - 0.8360309)  # layer 1 neuron 2
    h1_3 = relu(-1.7983123 * x0 - 0.13830443)  # layer 1 neuron 3
    h1_4 = relu(-0.05677128 * x0 - 0.7003182)  # layer 1 neuron 4
    h1_5 = relu(-1.3966193 * x0 - 1.8890872)  # layer 1 neuron 5
    h1_6 = relu(1.4848132 * x0 - 1.8450342)  # layer 1 neuron 6
    h1_7 = relu(0.156516 * x0 - 0.40392008)  # layer 1 neuron 7
    return -1.0628564 * h1_0 + 0.80665773 * h1_1 + 1.0920911 * h1_2 + 0.715485 * h1_3 + 0.1048713 * h1_4 + 0.924647 * h1_5 + 2.1861427 * h1_6 - 0.058465872 * h1_7 + 1.4523885
