# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(1.2861571 * x0 - 0.2025924)  # layer 1 neuron 0
    h1_1 = relu(-0.25142542 * x0 + 0.5066563)  # layer 1 neuron 1
    h1_2 = relu(0.5996773 * x0 + 0.34228536)  # layer 1 neuron 2
    h1_3 = relu(1.2808928 * x0 - 0.9458986)  # layer 1 neuron 3
    h1_4 = relu(0.68016285 * x0 + 1.0773231)  # layer 1 neuron 4
    h1_5 = relu(1.2600738 * x0 - 2.072738)  # layer 1 neuron 5
    h1_6 = relu(-0.047210723 * x0 - 0.21218982)  # layer 1 neuron 6
    h1_7 = relu(1.3930182 * x0 - 1.7117562)  # layer 1 neuron 7
    return 0.58621025 * h1_0 - 0.19339745 * h1_1 + 0.7897653 * h1_2 + 0.865253 * h1_3 + 0.44140866 * h1_4 + 1.5751262 * h1_5 + 0.067370534 * h1_6 + 1.1028 * h1_7 + 0.3493886
