# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.031146351 * x0 + 1.3264344)  # layer 1 neuron 0
    h1_1 = tanh(-0.02890613 * x0 - 1.6796975)  # layer 1 neuron 1
    h1_2 = tanh(-0.04217041 * x0 - 0.7933599)  # layer 1 neuron 2
    h1_3 = tanh(0.034853123 * x0 + 1.0578996)  # layer 1 neuron 3
    h1_4 = tanh(-0.031680882 * x0 - 1.2908891)  # layer 1 neuron 4
    h1_5 = tanh(0.028063176 * x0 + 1.9349886)  # layer 1 neuron 5
    h1_6 = tanh(0.82719094 * x0 + 1.5238495)  # layer 1 neuron 6
    h1_7 = tanh(-0.8661428 * x0 + 1.573067)  # layer 1 neuron 7
    return 1.314353 * h1_0 - 1.0012825 * h1_1 - 0.7623218 * h1_2 + 1.972797 * h1_3 - 2.037526 * h1_4 + 0.82877326 * h1_5 - 4.2108827 * h1_6 - 3.8124197 * h1_7 + 0.6715673
