# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = relu(1.3247235 * x0 + 0.024536084 * x1 + 0.93354356)  # layer 1 neuron 0
    h1_1 = relu(-0.11305903 * x0 - 1.555299 * x1 + 0.5284349)  # layer 1 neuron 1
    h1_2 = relu(-0.7206153 * x0 + 0.8284979 * x1 - 0.19518085)  # layer 1 neuron 2
    h1_3 = relu(1.3890876 * x0 + 0.23210998 * x1 + 0.23541625)  # layer 1 neuron 3
    h1_4 = relu(-0.21706235 * x0 + 0.077411935 * x1 - 0.15491727)  # layer 1 neuron 4
    h1_5 = relu(1.0298203 * x0 - 1.0084606 * x1 + 0.12471392)  # layer 1 neuron 5
    h1_6 = relu(-0.21581288 * x0 - 1.4954683 * x1 - 0.59318227)  # layer 1 neuron 6
    h1_7 = relu(-0.6971729 * x0 + 0.5927919 * x1 - 0.1973628)  # layer 1 neuron 7
    return 0.6920761 * h1_0 + 0.46073478 * h1_1 + 1.0035331 * h1_2 - 0.704615 * h1_3 + 0.15403572 * h1_4 + 0.7295423 * h1_5 - 0.56844676 * h1_6 + 0.42350107 * h1_7 - 0.011205347
