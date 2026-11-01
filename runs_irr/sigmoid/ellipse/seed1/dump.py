# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(0.05907885 * x0 + 0.0059370347 * x1 + 2.5330172)  # layer 1 neuron 0
    h1_1 = sigmoid(3.0051548 * x0 + 4.9235883 * x1 - 8.228363)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.5320709 * x0 + 7.936016 * x1 - 6.6017346)  # layer 1 neuron 2
    h1_3 = sigmoid(-3.969232 * x0 + 5.5563374 * x1 - 6.4786673)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.8339841 * x0 + 1.1696777 * x1 + 6.207134)  # layer 1 neuron 4
    h1_5 = sigmoid(-3.5643983 * x0 - 6.579976 * x1 - 6.6257863)  # layer 1 neuron 5
    h1_6 = sigmoid(5.254598 * x0 - 0.60732794 * x1 + 6.8132305)  # layer 1 neuron 6
    h1_7 = sigmoid(-3.5521514 * x0 + 0.31678212 * x1 - 4.2144976)  # layer 1 neuron 7
    h1_8 = sigmoid(0.056152817 * x0 + 0.0056958133 * x1 + 3.1114879)  # layer 1 neuron 8
    h1_9 = sigmoid(3.6757705 * x0 - 1.5828992 * x1 - 8.437611)  # layer 1 neuron 9
    h1_10 = sigmoid(-3.4649808 * x0 - 1.8032619 * x1 + 7.887308)  # layer 1 neuron 10
    h1_11 = sigmoid(2.5653634 * x0 - 6.2841535 * x1 - 8.237966)  # layer 1 neuron 11
    h1_12 = sigmoid(-0.7545024 * x0 - 7.791272 * x1 - 6.482346)  # layer 1 neuron 12
    h1_13 = sigmoid(5.4365625 * x0 + 2.620961 * x1 + 7.29902)  # layer 1 neuron 13
    h1_14 = sigmoid(1.0860282 * x0 + 8.385198 * x1 - 7.958523)  # layer 1 neuron 14
    h1_15 = sigmoid(-0.6199756 * x0 + 7.8706727 * x1 + 7.132511)  # layer 1 neuron 15
    return -1.531018 * h1_0 - 9.689275 * h1_1 - 9.4405155 * h1_2 - 8.675133 * h1_3 + 2.054388 * h1_4 - 8.602565 * h1_5 + 3.9208958 * h1_6 - 4.0265045 * h1_7 - 1.452014 * h1_8 - 9.223317 * h1_9 + 3.35538 * h1_10 - 8.658771 * h1_11 - 9.30376 * h1_12 + 4.457228 * h1_13 - 9.359893 * h1_14 + 5.1040325 * h1_15 - 1.3900688
