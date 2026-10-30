# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.8016033 * x0 + 0.8170014 * x1 - 1.8128108)  # layer 1 neuron 0
    h1_1 = sigmoid(1.6204972 * x0 - 1.6320822 * x1 - 5.9617715)  # layer 1 neuron 1
    h1_2 = sigmoid(1.9852419 * x0 + 2.0888739 * x1 + 1.1660379)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.182124 * x0 - 2.1159637 * x1 + 1.775315)  # layer 1 neuron 3
    h1_4 = sigmoid(1.0806949 * x0 - 0.51368874 * x1 + 3.0422227)  # layer 1 neuron 4
    h1_5 = sigmoid(1.2326386 * x0 + 1.305974 * x1 + 0.890561)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.5961317 * x0 - 1.4737474 * x1 + 0.849072)  # layer 1 neuron 6
    h1_7 = sigmoid(2.2893405 * x0 - 2.3319902 * x1 - 2.6499345)  # layer 1 neuron 7
    return 5.3964252 * h1_0 + 3.2116249 * h1_1 + 0.08786175 * h1_2 - 0.14700231 * h1_3 - 0.65052 * h1_4 + 0.15142281 * h1_5 + 0.30466944 * h1_6 + 2.712085 * h1_7 - 0.21055882
