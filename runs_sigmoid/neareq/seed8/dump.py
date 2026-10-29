# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(7.159262 * x0 - 7.1438417 * x1 + 1.9618747)  # layer 1 neuron 0
    h1_1 = sigmoid(9.08295 * x0 - 9.081426 * x1 - 1.9488703)  # layer 1 neuron 1
    h1_2 = sigmoid(-8.661259 * x0 + 8.6433 * x1 - 2.1050408)  # layer 1 neuron 2
    h1_3 = sigmoid(9.627383 * x0 - 9.62275 * x1 - 2.0248094)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.54294765 * x0 - 0.35681406 * x1 - 1.5092233)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.7125201 * x0 + 1.5218858 * x1 + 2.9823475)  # layer 1 neuron 5
    h1_6 = sigmoid(-5.815942 * x0 + 5.7988567 * x1 - 1.9028666)  # layer 1 neuron 6
    h1_7 = sigmoid(-8.948757 * x0 + 8.9304085 * x1 - 2.1363742)  # layer 1 neuron 7
    return 7.720133 * h1_0 - 14.668652 * h1_1 - 11.27843 * h1_2 - 13.706061 * h1_3 + 2.0808666 * h1_4 + 3.3370643 * h1_5 - 4.7295938 * h1_6 - 10.953535 * h1_7 + 2.8597574
