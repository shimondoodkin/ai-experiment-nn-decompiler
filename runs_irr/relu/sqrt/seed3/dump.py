# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-1.3013442 * x0 + 0.5501667)  # layer 1 neuron 0
    h1_1 = relu(-0.21999784 * x0 + 1.2151412)  # layer 1 neuron 1
    h1_2 = relu(-0.42831504 * x0 - 0.097937346)  # layer 1 neuron 2
    h1_3 = relu(-0.18355511 * x0 + 0.8777507)  # layer 1 neuron 3
    h1_4 = relu(-0.05677128 * x0 - 0.7003182)  # layer 1 neuron 4
    h1_5 = relu(-0.8797674 * x0 - 0.19706237)  # layer 1 neuron 5
    h1_6 = relu(0.92089987 * x0 - 0.3901025)  # layer 1 neuron 6
    h1_7 = relu(0.42276534 * x0 - 0.7092755)  # layer 1 neuron 7
    return -1.0410362 * h1_0 + 0.50874186 * h1_1 + 0.24559137 * h1_2 + 0.02058547 * h1_3 + 0.1048713 * h1_4 - 0.20169674 * h1_5 + 0.67052317 * h1_6 - 0.47748283 * h1_7 + 0.105056815
