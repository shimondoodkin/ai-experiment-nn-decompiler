# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(0.10619275 * x0 + 0.034711413 * x1 + 2.059293)  # layer 1 neuron 0
    h1_1 = tanh(-0.097558 * x0 - 0.09474709 * x1 + 2.2248003)  # layer 1 neuron 1
    h1_2 = tanh(-0.06102251 * x0 - 0.0025333497 * x1 - 1.8193926)  # layer 1 neuron 2
    h1_3 = tanh(-6.245161 * x0 + 6.261762 * x1 - 1.2560972)  # layer 1 neuron 3
    h1_4 = tanh(0.041964587 * x0 - 0.012702305 * x1 + 1.7491381)  # layer 1 neuron 4
    h1_5 = tanh(-0.09287194 * x0 - 0.092859425 * x1 + 2.1484022)  # layer 1 neuron 5
    h1_6 = tanh(0.01703467 * x0 - 0.023337938 * x1 + 1.7678183)  # layer 1 neuron 6
    h1_7 = tanh(-6.3650265 * x0 + 6.3797417 * x1 + 1.2639687)  # layer 1 neuron 7
    return -1.7769015 * h1_0 - 1.7839073 * h1_1 + 1.824469 * h1_2 - 11.852298 * h1_3 - 1.7747521 * h1_4 - 1.8475012 * h1_5 - 1.4369618 * h1_6 + 11.670899 * h1_7 - 1.4596305
