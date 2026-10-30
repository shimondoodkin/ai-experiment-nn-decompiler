# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-0.49780956 * x0 - 0.056662515 * x1 - 3.7696745)  # layer 1 neuron 0
    h1_1 = sigmoid(-8.67511 * x0 + 8.640343 * x1 - 2.0511606)  # layer 1 neuron 1
    h1_2 = sigmoid(-7.734279 * x0 + 7.7024317 * x1 - 1.9432104)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.230277 * x0 - 0.5648187 * x1 + 4.467041)  # layer 1 neuron 3
    h1_4 = sigmoid(0.4485553 * x0 - 0.21743765 * x1 + 4.058364)  # layer 1 neuron 4
    h1_5 = sigmoid(-11.590873 * x0 + 11.551746 * x1 + 2.1411004)  # layer 1 neuron 5
    h1_6 = sigmoid(-10.495026 * x0 + 10.458786 * x1 + 1.9463514)  # layer 1 neuron 6
    h1_7 = sigmoid(-8.600274 * x0 + 8.5657 * x1 - 2.042033)  # layer 1 neuron 7
    return 0.8703667 * h1_0 - 10.412445 * h1_1 - 7.669719 * h1_2 - 3.441138 * h1_3 - 3.2544954 * h1_4 + 11.479647 * h1_5 + 9.4412365 * h1_6 - 10.246778 * h1_7 - 2.2932637
