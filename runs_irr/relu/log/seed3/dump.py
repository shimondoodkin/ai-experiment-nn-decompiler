# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.52642643 * x0 + 0.808604)  # layer 1 neuron 0
    h1_1 = relu(-1.4649353 * x0 + 1.0613397)  # layer 1 neuron 1
    h1_2 = relu(-0.42831504 * x0 - 0.097937346)  # layer 1 neuron 2
    h1_3 = relu(-2.8979948 * x0 + 0.8527423)  # layer 1 neuron 3
    h1_4 = relu(-0.05677128 * x0 - 0.7003182)  # layer 1 neuron 4
    h1_5 = relu(-0.8797674 * x0 - 0.19706237)  # layer 1 neuron 5
    h1_6 = relu(0.8417596 * x0 - 0.25260392)  # layer 1 neuron 6
    h1_7 = relu(0.37213966 * x0 - 1.0759283)  # layer 1 neuron 7
    return -0.865491 * h1_0 - 0.7688663 * h1_1 + 0.24559137 * h1_2 - 1.3000456 * h1_3 + 0.1048713 * h1_4 - 0.20169674 * h1_5 + 0.54359996 * h1_6 - 0.5377082 * h1_7 - 0.09655232
