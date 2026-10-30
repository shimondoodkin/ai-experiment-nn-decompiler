# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(11.7249565 * x0 - 6.187423)  # layer 1 neuron 0
    h1_1 = sigmoid(-12.072777 * x0 + 6.334686)  # layer 1 neuron 1
    h1_2 = sigmoid(9.688683 * x0 - 5.2507477)  # layer 1 neuron 2
    h1_3 = sigmoid(-13.356571 * x0 - 6.654143)  # layer 1 neuron 3
    return -5.8080573 * h1_0 + 5.8185973 * h1_1 - 4.2747564 * h1_2 - 12.715849 * h1_3 + 0.787966
