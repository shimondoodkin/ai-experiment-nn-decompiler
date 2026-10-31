# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.9064035 * x0 + 1.1866221 * x1 - 3.8035734)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.90826786 * x0 - 0.9199875 * x1 - 3.4062312)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.0589563 * x0 - 1.069395 * x1 - 4.070551)  # layer 1 neuron 2
    h1_3 = sigmoid(0.8559054 * x0 + 0.858077 * x1 - 3.4109423)  # layer 1 neuron 3
    h1_4 = sigmoid(0.8913153 * x0 + 0.89996016 * x1 - 3.6218157)  # layer 1 neuron 4
    h1_5 = sigmoid(-1.4759829 * x0 + 1.6421698 * x1 + 5.9817743)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.3450263 * x0 - 1.3416297 * x1 - 1.8676091)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.7717546 * x0 - 0.8212487 * x1 + 2.093677)  # layer 1 neuron 7
    h1_8 = sigmoid(0.8334585 * x0 + 0.8531566 * x1 - 3.347055)  # layer 1 neuron 8
    h1_9 = sigmoid(-1.5756696 * x0 + 1.4565164 * x1 - 1.5922339)  # layer 1 neuron 9
    h1_10 = sigmoid(0.82411116 * x0 + 0.83069474 * x1 + 3.028677)  # layer 1 neuron 10
    h1_11 = sigmoid(-1.1752007 * x0 + 0.8595237 * x1 - 3.6233773)  # layer 1 neuron 11
    h1_12 = sigmoid(-1.2711036 * x0 - 1.2407851 * x1 - 4.9289894)  # layer 1 neuron 12
    h1_13 = sigmoid(0.752463 * x0 - 0.9964979 * x1 - 2.1426508)  # layer 1 neuron 13
    h1_14 = sigmoid(-0.778792 * x0 - 0.8146463 * x1 + 3.0404816)  # layer 1 neuron 14
    h1_15 = sigmoid(-1.0553335 * x0 + 0.80217654 * x1 + 3.086062)  # layer 1 neuron 15
    return -3.3189125 * h1_0 + 1.6867394 * h1_1 + 1.7750916 * h1_2 + 2.1332529 * h1_3 + 1.7502745 * h1_4 + 2.0586705 * h1_5 + 1.0103137 * h1_6 - 1.2293053 * h1_7 + 1.5784332 * h1_8 - 0.64231175 * h1_9 - 1.4738016 * h1_10 - 3.1277418 * h1_11 + 1.9653473 * h1_12 - 2.4440632 * h1_13 - 1.6242727 * h1_14 + 2.1257946 * h1_15 + 0.08167192
