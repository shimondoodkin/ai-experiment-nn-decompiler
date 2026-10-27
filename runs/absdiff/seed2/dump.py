# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = relu(0.66067654 * x0 - 0.6620454 * x1 + 0.009451158)  # layer 1 neuron 0
    h1_1 = relu(0.71168345 * x0 - 0.7117849 * x1 + 0.0010619428)  # layer 1 neuron 1
    h1_2 = relu(0.62829536 * x0 - 0.62836665 * x1 + 0.00087987224)  # layer 1 neuron 2
    h1_3 = relu(-0.7233756 * x0 + 0.72286445 * x1 + 0.041333202)  # layer 1 neuron 3
    h1_4 = relu(-0.00086424407 * x0 - 0.3762664 * x1 - 0.92989904)  # layer 1 neuron 4
    h1_5 = relu(0.13216703 * x0 - 0.13423645 * x1 + 0.80926865)  # layer 1 neuron 5
    h1_6 = relu(0.19941632 * x0 + 0.24865004 * x1 - 0.35042068)  # layer 1 neuron 6
    h1_7 = relu(-1.0018084 * x0 + 1.0005895 * x1 - 0.01293293)  # layer 1 neuron 7
    return 0.37917843 * h1_0 + 0.7214517 * h1_1 + 0.5058024 * h1_2 + 0.33551303 * h1_3 + 0.022398198 * h1_4 - 0.6175646 * h1_5 - 0.002131937 * h1_6 + 0.6743429 * h1_7 + 0.4947937
