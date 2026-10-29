# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = tanh(0.0000010756827 * x0 + 0.00000019628146)  # layer 1 neuron 0
    h1_1 = tanh(-0.0056953235 * x0 + 0.5666201)  # layer 1 neuron 1
    h1_2 = tanh(0.016279815 * x0 - 0.15154566)  # layer 1 neuron 2
    h1_3 = tanh(-0.0033923702 * x0 - 1.1723872)  # layer 1 neuron 3
    return 0.00000020134628 * h1_0 - 0.61142105 * h1_1 + 0.08346366 * h1_2 + 0.33928692 * h1_3 - 0.47468522
