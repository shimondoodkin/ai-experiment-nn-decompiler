# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.99147296 * x0 + 0.18884242)  # layer 1 neuron 0
    h1_1 = relu(-0.5797458 * x0 + 1.0847921)  # layer 1 neuron 1
    h1_2 = relu(-0.42831504 * x0 - 0.097937346)  # layer 1 neuron 2
    h1_3 = relu(-1.0560175 * x0 + 1.0252597)  # layer 1 neuron 3
    h1_4 = relu(-0.05677128 * x0 - 0.7003182)  # layer 1 neuron 4
    h1_5 = relu(-0.8797674 * x0 - 0.19706237)  # layer 1 neuron 5
    h1_6 = relu(0.76090336 * x0 - 0.54326725)  # layer 1 neuron 6
    h1_7 = relu(0.7244708 * x0 - 0.3628892)  # layer 1 neuron 7
    return -0.22938174 * h1_0 + 0.70077235 * h1_1 + 0.24559137 * h1_2 + 1.2452551 * h1_3 + 0.1048713 * h1_4 - 0.20169674 * h1_5 + 0.44295707 * h1_6 - 0.6332497 * h1_7 + 0.738075
