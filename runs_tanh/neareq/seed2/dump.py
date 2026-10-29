# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(0.25144282 * x0 - 0.22024551 * x1 + 2.1451223)  # layer 1 neuron 0
    h1_1 = tanh(-0.18446663 * x0 + 0.2527767 * x1 - 2.1104152)  # layer 1 neuron 1
    h1_2 = tanh(-0.16825615 * x0 + 0.24875945 * x1 - 2.2497563)  # layer 1 neuron 2
    h1_3 = tanh(0.03193845 * x0 - 0.30100888 * x1 + 2.2057662)  # layer 1 neuron 3
    h1_4 = tanh(6.7674136 * x0 - 6.7507334 * x1 - 1.2680789)  # layer 1 neuron 4
    h1_5 = tanh(-3.6566317 * x0 + 3.684238 * x1 - 0.9472197)  # layer 1 neuron 5
    h1_6 = tanh(-5.7483726 * x0 + 5.7662373 * x1 - 1.2412845)  # layer 1 neuron 6
    h1_7 = tanh(0.29676455 * x0 - 0.1452541 * x1 + 2.0769358)  # layer 1 neuron 7
    return -2.3794599 * h1_0 + 2.3328867 * h1_1 + 2.4909937 * h1_2 - 2.4119937 * h1_3 - 11.718541 * h1_4 - 4.4991198 * h1_5 - 10.5180435 * h1_6 - 2.4463131 * h1_7 - 1.5187378
