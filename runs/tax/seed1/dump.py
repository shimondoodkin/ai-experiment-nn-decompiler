# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.65175736 * x0 - 1.3040916)  # layer 1 neuron 0
    h1_1 = relu(-0.44137824 * x0 - 0.12244415)  # layer 1 neuron 1
    h1_2 = relu(-0.050348084 * x0 + 3.928657)  # layer 1 neuron 2
    h1_3 = relu(0.6180859 * x0 - 1.2702804)  # layer 1 neuron 3
    h1_4 = relu(-0.024256695 * x0 + 4.763882)  # layer 1 neuron 4
    h1_5 = relu(0.40831408 * x0 - 0.362695)  # layer 1 neuron 5
    h1_6 = relu(-0.20572531 * x0 - 0.07290876)  # layer 1 neuron 6
    h1_7 = relu(0.3115829 * x0 - 0.50528973)  # layer 1 neuron 7
    return 0.22146748 * h1_0 - 0.0014123917 * h1_1 + 4.2832613 * h1_2 + 0.28131846 * h1_3 - 3.5517135 * h1_4 - 0.030627983 * h1_5 - 0.059261084 * h1_6 + 0.0273881 * h1_7 - 0.39553866
