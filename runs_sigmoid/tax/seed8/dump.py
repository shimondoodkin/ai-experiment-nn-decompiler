# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.07010213 * x0 - 6.346153)  # layer 1 neuron 0
    h1_1 = sigmoid(0.08856326 * x0 - 14.825473)  # layer 1 neuron 1
    h1_2 = sigmoid(0.07399384 * x0 - 2.5409224)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.09710756 * x0 + 8.955715)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.110012785 * x0 - 0.8185087)  # layer 1 neuron 4
    h1_5 = sigmoid(0.07994773 * x0 - 9.786836)  # layer 1 neuron 5
    h1_6 = sigmoid(0.08854694 * x0 - 16.12243)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.089164056 * x0 + 12.255866)  # layer 1 neuron 7
    return 10.670321 * h1_0 + 10.245569 * h1_1 + 10.407216 * h1_2 - 4.8924994 * h1_3 - 0.06840686 * h1_4 + 10.459146 * h1_5 + 10.157784 * h1_6 - 6.3650336 * h1_7 + 10.651037
