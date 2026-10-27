# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = relu(0.51201534 * x0 - 0.5060016 * x1 + 0.068984285)  # layer 1 neuron 0
    h1_1 = relu(0.018649092 * x0 - 1.2350698 * x1 + 0.6527775)  # layer 1 neuron 1
    h1_2 = relu(-0.7444662 * x0 + 0.7464684 * x1 + 0.004190557)  # layer 1 neuron 2
    h1_3 = relu(0.0033064687 * x0 + 0.25558156 * x1 - 0.6284281)  # layer 1 neuron 3
    h1_4 = relu(-0.03357309 * x0 + 0.033694126 * x1 + 0.00001967177)  # layer 1 neuron 4
    h1_5 = relu(1.1266874 * x0 - 1.1246046 * x1 + 0.013966587)  # layer 1 neuron 5
    h1_6 = relu(0.015088952 * x0 - 0.69029003 * x1 + 0.3386915)  # layer 1 neuron 6
    h1_7 = relu(-0.65129673 * x0 + 0.6541506 * x1 + 0.038553372)  # layer 1 neuron 7
    return 0.14016996 * h1_0 + 0.07069334 * h1_1 + 0.9529933 * h1_2 - 0.08087148 * h1_3 + 0.028029786 * h1_4 + 0.8245518 * h1_5 - 0.123608656 * h1_6 + 0.44483432 * h1_7 - 0.025997205
