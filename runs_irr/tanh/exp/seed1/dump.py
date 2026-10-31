# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.060948737 * x0 + 1.2778413)  # layer 1 neuron 0
    h1_1 = tanh(-0.058814168 * x0 - 1.3691909)  # layer 1 neuron 1
    h1_2 = tanh(0.07536474 * x0 + 1.0900478)  # layer 1 neuron 2
    h1_3 = tanh(0.8133715 * x0 - 0.6029045)  # layer 1 neuron 3
    h1_4 = tanh(-1.2035 * x0 + 1.941695)  # layer 1 neuron 4
    h1_5 = tanh(-0.060997795 * x0 - 1.4759252)  # layer 1 neuron 5
    h1_6 = tanh(-0.066013455 * x0 - 1.1301848)  # layer 1 neuron 6
    h1_7 = tanh(1.8627019 * x0 - 3.8822582)  # layer 1 neuron 7
    return 0.8948719 * h1_0 - 0.8945071 * h1_1 + 1.5525022 * h1_2 + 1.25223 * h1_3 - 1.7966784 * h1_4 - 1.5961692 * h1_5 - 1.0751463 * h1_6 + 2.3679397 * h1_7 + 0.65824723
