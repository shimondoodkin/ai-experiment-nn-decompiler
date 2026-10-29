# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(0.1240633 * x0 - 0.04543848 * x1 + 1.8683441)  # layer 1 neuron 0
    h1_1 = tanh(0.10080172 * x0 - 0.068331376 * x1 + 1.9149671)  # layer 1 neuron 1
    h1_2 = tanh(-6.8880982 * x0 + 6.884908 * x1 + 1.3238795)  # layer 1 neuron 2
    h1_3 = tanh(-6.2642107 * x0 + 6.254124 * x1 - 1.2753283)  # layer 1 neuron 3
    h1_4 = tanh(-0.015286752 * x0 - 0.049781978 * x1 + 0.18500075)  # layer 1 neuron 4
    h1_5 = tanh(0.0545902 * x0 + 0.1551242 * x1 - 1.8997715)  # layer 1 neuron 5
    h1_6 = tanh(-0.04336431 * x0 + 0.100643754 * x1 - 1.8219378)  # layer 1 neuron 6
    h1_7 = tanh(-0.1383918 * x0 + 0.041576173 * x1 - 1.649837)  # layer 1 neuron 7
    return -2.3291981 * h1_0 - 2.1461663 * h1_1 + 10.92958 * h1_2 - 11.7136 * h1_3 - 0.10093871 * h1_4 + 1.998914 * h1_5 + 2.1911304 * h1_6 + 2.0427074 * h1_7 - 0.9051256
