# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-3.0122714 * x0 - 2.486009)  # layer 1 neuron 0
    h1_1 = sigmoid(0.4034536 * x0 + 0.7422965)  # layer 1 neuron 1
    h1_2 = sigmoid(-2.2864628 * x0 - 2.228818)  # layer 1 neuron 2
    h1_3 = sigmoid(0.9727104 * x0 - 0.1693516)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.541382 * x0 - 4.8202233)  # layer 1 neuron 4
    h1_5 = sigmoid(-2.8733459 * x0 - 5.6283565)  # layer 1 neuron 5
    h1_6 = sigmoid(2.8827748 * x0 - 5.390024)  # layer 1 neuron 6
    h1_7 = sigmoid(-3.2731457 * x0 + 2.8511188)  # layer 1 neuron 7
    return 0.8276874 * h1_0 + 0.68786657 * h1_1 + 1.1544935 * h1_2 + 0.7116253 * h1_3 + 2.628675 * h1_4 + 2.449359 * h1_5 + 4.592398 * h1_6 - 1.1709895 * h1_7 + 0.09292552
