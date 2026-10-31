# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-1.3249216 * x0 - 1.3686026 * x1 + 5.0698686)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.0879959 * x0 + 1.0449775 * x1 - 4.303012)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.77434254 * x0 + 0.8150292 * x1 - 2.9801903)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.7996464 * x0 + 0.83820826 * x1 - 3.1428983)  # layer 1 neuron 3
    h1_4 = sigmoid(0.7253104 * x0 - 0.7381954 * x1 + 2.3636148)  # layer 1 neuron 4
    h1_5 = sigmoid(1.0515568 * x0 - 1.1660198 * x1 - 4.2590423)  # layer 1 neuron 5
    h1_6 = sigmoid(0.7258331 * x0 - 0.7362235 * x1 + 2.2435403)  # layer 1 neuron 6
    h1_7 = sigmoid(1.2326174 * x0 + 1.2565126 * x1 - 4.5880365)  # layer 1 neuron 7
    h1_8 = sigmoid(1.7230387 * x0 + 1.738072 * x1 + 6.4023714)  # layer 1 neuron 8
    h1_9 = sigmoid(0.9994648 * x0 - 1.0078865 * x1 - 3.7567637)  # layer 1 neuron 9
    h1_10 = sigmoid(-1.5185364 * x0 + 1.5411614 * x1 + 1.7511097)  # layer 1 neuron 10
    h1_11 = sigmoid(-1.1097217 * x0 - 1.1338552 * x1 - 2.4331732)  # layer 1 neuron 11
    h1_12 = sigmoid(-1.416726 * x0 - 1.4703096 * x1 + 2.0965452)  # layer 1 neuron 12
    h1_13 = sigmoid(1.0715345 * x0 + 0.99885285 * x1 - 2.3211608)  # layer 1 neuron 13
    h1_14 = sigmoid(-0.9371596 * x0 + 0.8930055 * x1 + 3.3215075)  # layer 1 neuron 14
    h1_15 = sigmoid(-0.92770165 * x0 + 0.88853496 * x1 + 3.2823715)  # layer 1 neuron 15
    return -2.2505383 * h1_0 - 2.0180707 * h1_1 - 2.0916042 * h1_2 - 1.6596307 * h1_3 + 0.99479634 * h1_4 - 1.8802437 * h1_5 + 1.4071835 * h1_6 + 2.4660797 * h1_7 - 2.9352803 * h1_8 - 1.9024959 * h1_9 + 0.6611378 * h1_10 + 2.7957067 * h1_11 - 0.81321996 * h1_12 + 0.91856396 * h1_13 + 1.5537908 * h1_14 + 1.6464379 * h1_15 - 0.005634157
