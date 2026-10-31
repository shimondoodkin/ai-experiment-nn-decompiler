# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.84104 * x0 - 0.5867781)  # layer 1 neuron 0
    h1_1 = tanh(-0.0801038 * x0 - 2.1256502)  # layer 1 neuron 1
    h1_2 = tanh(1.3484515 * x0 - 2.8595846)  # layer 1 neuron 2
    h1_3 = tanh(0.10030604 * x0 + 1.1205432)  # layer 1 neuron 3
    h1_4 = tanh(-0.08735419 * x0 - 1.315784)  # layer 1 neuron 4
    h1_5 = tanh(0.07880683 * x0 + 2.3058507)  # layer 1 neuron 5
    h1_6 = tanh(-1.0983824 * x0 + 1.8224263)  # layer 1 neuron 6
    h1_7 = tanh(-1.7001828 * x0 + 3.5726693)  # layer 1 neuron 7
    return 1.1526967 * h1_0 - 1.1958845 * h1_1 + 1.4634516 * h1_2 + 1.8449215 * h1_3 - 1.8045388 * h1_4 + 1.0121156 * h1_5 - 1.58154 * h1_6 - 1.5987513 * h1_7 + 0.9464216
