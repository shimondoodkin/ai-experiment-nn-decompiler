# dumped from a neural network: 2 inputs -> 8 -> 3 outputs
# inputs: x0: float [1, 6.9], x1: float [0.1, 2.5]
# output: argmax over 3 outputs: the returned class index
def net(x0, x1):
    h1_0 = sigmoid(-0.7897094 * x0 - 1.0970942 * x1 + 5.530363)  # layer 1 neuron 0
    h1_1 = sigmoid(0.60197777 * x0 + 1.1846247 * x1 - 4.045535)  # layer 1 neuron 1
    h1_2 = sigmoid(1.6305939 * x0 + 3.5524032 * x1 - 6.347963)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.7850122 * x0 - 0.7580271 * x1 + 4.740742)  # layer 1 neuron 3
    h1_4 = sigmoid(-1.026726 * x0 - 0.16290486 * x1 + 5.3701844)  # layer 1 neuron 4
    h1_5 = sigmoid(0.68676096 * x0 + 1.1968019 * x1 - 5.151463)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.74948484 * x0 - 1.2473679 * x1 + 3.9283564)  # layer 1 neuron 6
    h1_7 = sigmoid(0.9625947 * x0 + 4.031685 * x1 - 4.9917965)  # layer 1 neuron 7
    o0 = 2.777276 * h1_0 - 3.5951486 * h1_1 - 4.9561057 * h1_2 + 2.3579123 * h1_3 + 2.925878 * h1_4 - 2.7767966 * h1_5 + 4.170731 * h1_6 - 5.3865185 * h1_7 + 1.6998656  # output 0
    o1 = 2.562424 * h1_0 - 1.3950106 * h1_1 + 1.9859755 * h1_2 + 1.6124157 * h1_3 + 2.712027 * h1_4 - 3.9871616 * h1_5 - 1.0648751 * h1_6 + 1.3737428 * h1_7 - 1.3730383  # output 1
    o2 = -3.6509495 * h1_0 + 3.291002 * h1_1 + 1.107042 * h1_2 - 3.3054783 * h1_3 - 4.834962 * h1_4 + 4.869031 * h1_5 - 4.5411844 * h1_6 + 1.576393 * h1_7 + 0.21895145  # output 2
    return argmax(o0, o1, o2)
