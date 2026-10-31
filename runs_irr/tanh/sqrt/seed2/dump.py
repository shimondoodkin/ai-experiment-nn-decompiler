# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.5364981 * x0 + 0.083001226)  # layer 1 neuron 0
    h1_1 = tanh(-1.1676227 * x0 - 0.58644205)  # layer 1 neuron 1
    h1_2 = tanh(0.18660465 * x0 - 0.34958678)  # layer 1 neuron 2
    h1_3 = tanh(0.20095505 * x0 + 0.05843448)  # layer 1 neuron 3
    h1_4 = tanh(0.16165955 * x0 - 0.42710334)  # layer 1 neuron 4
    h1_5 = tanh(3.4295301 * x0 + 0.4186376)  # layer 1 neuron 5
    h1_6 = tanh(-0.18115455 * x0 + 0.48979685)  # layer 1 neuron 6
    h1_7 = tanh(-0.24580505 * x0 + 0.7954323)  # layer 1 neuron 7
    return 0.4167116 * h1_0 - 0.42938748 * h1_1 + 0.2101807 * h1_2 + 0.6467568 * h1_3 + 0.07684142 * h1_4 + 0.37363374 * h1_5 - 0.47014704 * h1_6 - 0.15961494 * h1_7 + 0.10103333
