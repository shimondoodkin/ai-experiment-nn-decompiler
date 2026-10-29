# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-1.0439104 * x0 - 2.2896435 * x1 + 1.7438531)  # layer 1 neuron 0
    h1_1 = sigmoid(0.9485384 * x0 - 0.91879827 * x1 - 2.1208003)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.3809023 * x0 - 1.7810236 * x1 + 1.6301823)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.95575243 * x0 + 0.9561798 * x1 - 2.126789)  # layer 1 neuron 3
    h1_4 = sigmoid(1.0136988 * x0 + 1.3877349 * x1 + 1.652956)  # layer 1 neuron 4
    h1_5 = sigmoid(0.41863653 * x0 + 1.2284486 * x1 - 0.014353671)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.209339 * x0 - 1.584879 * x1 + 1.4252614)  # layer 1 neuron 6
    h1_7 = sigmoid(1.1013576 * x0 + 1.4250717 * x1 + 1.2780644)  # layer 1 neuron 7
    return -0.8691642 * h1_0 + 5.744267 * h1_1 + 0.35677847 * h1_2 + 5.7169614 * h1_3 + 0.2647976 * h1_4 - 0.76469576 * h1_5 + 0.23691258 * h1_6 + 0.091107614 * h1_7 - 0.44199234
