# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(0.60991824 * x0 - 0.05882649 * x1 + 3.0082078)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.20419079 * x0 - 0.70128834 * x1 + 3.831982)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.5207617 * x0 + 0.1417692 * x1 - 2.374078)  # layer 1 neuron 2
    h1_3 = sigmoid(-11.391683 * x0 + 11.432839 * x1 - 2.4444447)  # layer 1 neuron 3
    h1_4 = sigmoid(0.76856107 * x0 - 0.6093585 * x1 + 0.87131363)  # layer 1 neuron 4
    h1_5 = sigmoid(0.07886289 * x0 - 0.47804934 * x1 + 3.0389106)  # layer 1 neuron 5
    h1_6 = sigmoid(0.26835608 * x0 - 0.45068708 * x1 + 2.0882733)  # layer 1 neuron 6
    h1_7 = sigmoid(-12.280662 * x0 + 12.273867 * x1 + 2.4091175)  # layer 1 neuron 7
    return -1.3107773 * h1_0 - 1.4504701 * h1_1 + 0.66854167 * h1_2 - 16.519886 * h1_3 - 1.2155019 * h1_4 - 1.4078876 * h1_5 - 1.3038645 * h1_6 + 14.458104 * h1_7 - 0.85005003
