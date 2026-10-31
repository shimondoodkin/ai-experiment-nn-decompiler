# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(1.0376977 * x0 - 0.53258485)  # layer 1 neuron 0
    h1_1 = sigmoid(1.5064617 * x0 - 2.2000182)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.5386034 * x0 + 2.2801762)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.5979712 * x0 + 2.5378385)  # layer 1 neuron 3
    h1_4 = sigmoid(2.4543502 * x0 - 5.217874)  # layer 1 neuron 4
    h1_5 = sigmoid(1.5557116 * x0 - 2.545415)  # layer 1 neuron 5
    h1_6 = sigmoid(3.7876453 * x0 - 7.837134)  # layer 1 neuron 6
    h1_7 = sigmoid(1.69986 * x0 - 3.0491104)  # layer 1 neuron 7
    return 1.4956461 * h1_0 + 1.3573273 * h1_1 - 0.49115866 * h1_2 - 0.38237923 * h1_3 + 3.091333 * h1_4 + 1.8820833 * h1_5 + 2.6802292 * h1_6 + 1.5119693 * h1_7 + 0.8900325
