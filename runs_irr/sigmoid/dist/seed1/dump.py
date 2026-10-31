# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(0.53998125 * x0 - 1.9172885 * x1 - 2.8094068)  # layer 1 neuron 0
    h1_1 = sigmoid(0.12939991 * x0 + 1.2602684 * x1 - 3.5113704)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.564087 * x0 + 1.2500023 * x1 - 1.9959141)  # layer 1 neuron 2
    h1_3 = sigmoid(0.6946706 * x0 + 1.9314404 * x1 - 1.6738366)  # layer 1 neuron 3
    h1_4 = sigmoid(0.12548295 * x0 - 0.37473702 * x1 + 0.51325583)  # layer 1 neuron 4
    h1_5 = sigmoid(1.8349074 * x0 - 0.038072497 * x1 - 4.396674)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.4345284 * x0 - 0.046171643 * x1 - 0.06182629)  # layer 1 neuron 6
    h1_7 = sigmoid(0.06853511 * x0 - 1.3654742 * x1 - 3.7598104)  # layer 1 neuron 7
    h1_8 = sigmoid(1.3456633 * x0 + 0.16932833 * x1 + 1.3862238)  # layer 1 neuron 8
    h1_9 = sigmoid(2.411708 * x0 - 0.048954774 * x1 - 1.7589542)  # layer 1 neuron 9
    h1_10 = sigmoid(0.10299355 * x0 - 0.2905142 * x1 + 0.33376113)  # layer 1 neuron 10
    h1_11 = sigmoid(0.82931054 * x0 - 3.3646414 * x1 - 1.6388208)  # layer 1 neuron 11
    h1_12 = sigmoid(-1.7908388 * x0 - 1.3063481 * x1 - 1.9721075)  # layer 1 neuron 12
    h1_13 = sigmoid(-0.013639462 * x0 - 0.30110177 * x1 + 0.07089121)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.1685233 * x0 - 0.03409577 * x1 - 3.05842)  # layer 1 neuron 14
    h1_15 = sigmoid(-0.45526236 * x0 - 0.66863346 * x1 - 0.7941591)  # layer 1 neuron 15
    return 0.86352336 * h1_0 + 2.0591774 * h1_1 + 1.0997415 * h1_2 + 1.2157634 * h1_3 - 0.08185146 * h1_4 + 1.6309801 * h1_5 + 0.11848761 * h1_6 + 1.6865549 * h1_7 - 0.3160627 * h1_8 + 1.1101253 * h1_9 - 0.058085293 * h1_10 + 0.50482017 * h1_11 + 0.9039987 * h1_12 - 0.024943957 * h1_13 + 1.8384602 * h1_14 + 0.2184803 * h1_15 - 0.33721736
