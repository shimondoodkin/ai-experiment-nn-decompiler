# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = tanh(-0.015404745 * x0 - 0.023808718 * x1 + 1.2694436)  # layer 1 neuron 0
    h1_1 = tanh(0.02144565 * x0 + 0.030097997 * x1 - 1.3365957)  # layer 1 neuron 1
    h1_2 = tanh(0.014730033 * x0 + 0.023361968 * x1 - 1.1983639)  # layer 1 neuron 2
    h1_3 = tanh(0.38430732 * x0 - 0.3782352 * x1 + 0.79996055)  # layer 1 neuron 3
    h1_4 = tanh(-0.09399711 * x0 - 0.091604464 * x1 - 1.4694228)  # layer 1 neuron 4
    h1_5 = tanh(-0.031926114 * x0 - 0.02375233 * x1 - 0.80307746)  # layer 1 neuron 5
    h1_6 = tanh(0.037475582 * x0 + 0.047150232 * x1 - 1.1160344)  # layer 1 neuron 6
    h1_7 = tanh(-0.38143086 * x0 + 0.38741243 * x1 + 0.80412215)  # layer 1 neuron 7
    return 0.90038735 * h1_0 - 0.75941896 * h1_1 - 0.80066246 * h1_2 - 3.750011 * h1_3 - 0.9348898 * h1_4 - 0.9052237 * h1_5 - 1.5764955 * h1_6 - 3.70987 * h1_7 + 0.6397758
