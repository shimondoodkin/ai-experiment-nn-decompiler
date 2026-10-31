# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.7639644 * x0 + 0.3297381)  # layer 1 neuron 0
    h1_1 = relu(-0.44137824 * x0 - 0.12244415)  # layer 1 neuron 1
    h1_2 = relu(0.051719323 * x0 + 0.47711027)  # layer 1 neuron 2
    h1_3 = relu(0.71547854 * x0 + 0.24679135)  # layer 1 neuron 3
    h1_4 = relu(-1.4006516 * x0 + 0.5719502)  # layer 1 neuron 4
    h1_5 = relu(0.5503931 * x0 - 0.9078868)  # layer 1 neuron 5
    h1_6 = relu(-0.20572531 * x0 - 0.07290876)  # layer 1 neuron 6
    h1_7 = relu(0.34755683 * x0 - 0.14099564)  # layer 1 neuron 7
    return 0.2832194 * h1_0 - 0.0014123917 * h1_1 + 0.440743 * h1_2 + 0.34412208 * h1_3 - 0.5447366 * h1_4 - 0.3767947 * h1_5 - 0.059261084 * h1_6 + 0.06433824 * h1_7 + 0.096517645
