# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.023821566 * x0 - 1.722137)  # layer 1 neuron 0
    h1_1 = tanh(-0.04328664 * x0 + 4.689682)  # layer 1 neuron 1
    h1_2 = tanh(0.11702973 * x0 + 3.6475513)  # layer 1 neuron 2
    h1_3 = tanh(0.042397235 * x0 - 6.3158627)  # layer 1 neuron 3
    h1_4 = tanh(-2.3147597 * x0 - 3.8558972)  # layer 1 neuron 4
    h1_5 = tanh(0.4844745 * x0 + 3.9579926)  # layer 1 neuron 5
    h1_6 = tanh(-0.018410565 * x0 + 0.0034786111)  # layer 1 neuron 6
    h1_7 = tanh(0.048835784 * x0 - 8.966335)  # layer 1 neuron 7
    return 6.9926057 * h1_0 - 7.011226 * h1_1 + 6.790342 * h1_2 + 7.309441 * h1_3 - 7.1505814 * h1_4 + 6.8072743 * h1_5 - 7.0219073 * h1_6 + 7.244985 * h1_7 + 6.9047976
