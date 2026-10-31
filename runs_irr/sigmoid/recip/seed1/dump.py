# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.046979256 * x0 + 0.9763743)  # layer 1 neuron 0
    h1_1 = sigmoid(-2.6961331 * x0 + 0.69756985)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.8261901 * x0 + 0.6022941)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.014593641 * x0 + 0.73726743)  # layer 1 neuron 3
    h1_4 = sigmoid(-6.2557783 * x0 + 2.224564)  # layer 1 neuron 4
    h1_5 = sigmoid(1.1259831 * x0 - 0.7232669)  # layer 1 neuron 5
    h1_6 = sigmoid(-2.3039415 * x0 + 0.8836255)  # layer 1 neuron 6
    h1_7 = sigmoid(0.7024026 * x0 - 0.40812597)  # layer 1 neuron 7
    return 0.22432196 * h1_0 + 1.0326874 * h1_1 + 0.8170641 * h1_2 + 0.27303627 * h1_3 + 1.6185392 * h1_4 - 0.24310768 * h1_5 + 0.7740186 * h1_6 - 0.08841891 * h1_7 + 0.18471792
