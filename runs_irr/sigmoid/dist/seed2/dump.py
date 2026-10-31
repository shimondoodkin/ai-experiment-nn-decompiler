# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(0.547428 * x0 - 3.3115811 * x1 - 2.1108634)  # layer 1 neuron 0
    h1_1 = sigmoid(0.6655944 * x0 - 0.4635449 * x1 + 0.37982115)  # layer 1 neuron 1
    h1_2 = sigmoid(1.4096298 * x0 - 2.01878 * x1 - 2.0234947)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.2591449 * x0 - 1.2906728 * x1 - 1.6893553)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.048677444 * x0 - 0.476729 * x1 - 0.79129)  # layer 1 neuron 4
    h1_5 = sigmoid(1.1963872 * x0 + 0.228576 * x1 - 3.1839523)  # layer 1 neuron 5
    h1_6 = sigmoid(0.45824698 * x0 + 1.913467 * x1 - 1.4196503)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.8380579 * x0 + 0.6770852 * x1 - 1.5692569)  # layer 1 neuron 7
    h1_8 = sigmoid(0.3370528 * x0 - 1.6253213 * x1 - 4.0716977)  # layer 1 neuron 8
    h1_9 = sigmoid(0.2209839 * x0 + 1.4624842 * x1 - 3.8437312)  # layer 1 neuron 9
    h1_10 = sigmoid(0.41083118 * x0 - 1.2288123 * x1 - 1.6741643)  # layer 1 neuron 10
    h1_11 = sigmoid(-0.05807324 * x0 - 0.39447957 * x1 - 0.51374054)  # layer 1 neuron 11
    h1_12 = sigmoid(0.52532375 * x0 - 2.1573968 * x1 - 1.8367184)  # layer 1 neuron 12
    h1_13 = sigmoid(-2.1239676 * x0 - 0.5760865 * x1 + 1.597031)  # layer 1 neuron 13
    h1_14 = sigmoid(0.24257532 * x0 - 0.6040442 * x1 - 0.1762271)  # layer 1 neuron 14
    h1_15 = sigmoid(-1.2601293 * x0 + 0.2223584 * x1 - 3.3800392)  # layer 1 neuron 15
    return 0.30303088 * h1_0 - 0.15078641 * h1_1 + 0.7023717 * h1_2 + 1.1250743 * h1_3 + 0.18044305 * h1_4 + 2.1013644 * h1_5 + 1.195595 * h1_6 + 1.3328094 * h1_7 + 1.7007002 * h1_8 + 1.711655 * h1_9 + 0.12466766 * h1_10 + 0.19001187 * h1_11 + 0.2856918 * h1_12 - 1.0141873 * h1_13 - 0.021130247 * h1_14 + 2.1263313 * h1_15 + 0.1441498
