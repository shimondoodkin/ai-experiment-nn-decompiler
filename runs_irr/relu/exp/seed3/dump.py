# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.83881354 * x0 + 1.6799122)  # layer 1 neuron 0
    h1_1 = relu(0.3082024 * x0 + 1.4470204)  # layer 1 neuron 1
    h1_2 = relu(-1.5797671 * x0 + 1.9362952)  # layer 1 neuron 2
    h1_3 = relu(0.26578555 * x0 + 1.3177574)  # layer 1 neuron 3
    h1_4 = relu(-0.05677128 * x0 - 0.7003182)  # layer 1 neuron 4
    h1_5 = relu(-0.83787775 * x0 + 1.6779361)  # layer 1 neuron 5
    h1_6 = relu(1.3208901 * x0 - 0.12817568)  # layer 1 neuron 6
    h1_7 = relu(0.15425293 * x0 - 0.40046644)  # layer 1 neuron 7
    return -1.7012249 * h1_0 + 1.0985249 * h1_1 + 1.955285 * h1_2 + 0.86443734 * h1_3 + 0.1048713 * h1_4 - 1.8291994 * h1_5 + 1.1961968 * h1_6 - 0.05652732 * h1_7 + 0.2748338
