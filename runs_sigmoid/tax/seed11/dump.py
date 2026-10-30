# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.09166731 * x0 + 8.740189)  # layer 1 neuron 0
    h1_1 = sigmoid(0.02546516 * x0 + 0.46866304)  # layer 1 neuron 1
    h1_2 = sigmoid(0.08777814 * x0 - 15.768684)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.077925496 * x0 + 13.76672)  # layer 1 neuron 3
    h1_4 = sigmoid(0.07980756 * x0 - 11.734436)  # layer 1 neuron 4
    h1_5 = sigmoid(0.08152158 * x0 - 9.989517)  # layer 1 neuron 5
    h1_6 = sigmoid(0.05682331 * x0 - 2.5141141)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.09445075 * x0 + 8.9619875)  # layer 1 neuron 7
    return -5.93151 * h1_0 + 10.50249 * h1_1 + 10.562814 * h1_2 - 6.144606 * h1_3 + 10.521968 * h1_4 + 10.46868 * h1_5 + 10.466487 * h1_6 - 5.87669 * h1_7 + 10.485609
