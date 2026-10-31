# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(2.5918543 * x0 - 1.5970118 * x1 - 1.5445244)  # layer 1 neuron 0
    h1_1 = sigmoid(0.103245445 * x0 + 2.1477938 * x1 + 3.1147408)  # layer 1 neuron 1
    h1_2 = sigmoid(0.6341456 * x0 + 0.8733372 * x1 - 0.96216)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.0188028 * x0 - 0.60501903 * x1 - 1.3942171)  # layer 1 neuron 3
    h1_4 = sigmoid(1.942656 * x0 - 2.364056 * x1 + 1.6548554)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.8680805 * x0 - 0.7703052 * x1 - 1.6228334)  # layer 1 neuron 5
    h1_6 = sigmoid(0.14592019 * x0 + 2.7823553 * x1 - 1.3826889)  # layer 1 neuron 6
    h1_7 = sigmoid(2.1601126 * x0 + 2.2004766 * x1 - 1.5060533)  # layer 1 neuron 7
    h1_8 = sigmoid(0.9628287 * x0 - 2.7795312 * x1 - 1.5987868)  # layer 1 neuron 8
    h1_9 = sigmoid(-0.84964895 * x0 + 1.1293741 * x1 - 2.3826835)  # layer 1 neuron 9
    h1_10 = sigmoid(-3.1157331 * x0 + 0.64578426 * x1 - 1.6046822)  # layer 1 neuron 10
    h1_11 = sigmoid(1.1974643 * x0 + 2.5912313 * x1 + 1.6038557)  # layer 1 neuron 11
    h1_12 = sigmoid(2.6599348 * x0 + 1.4021533 * x1 + 1.5284828)  # layer 1 neuron 12
    h1_13 = sigmoid(-1.0483617 * x0 + 0.177659 * x1 + 1.3417884)  # layer 1 neuron 13
    h1_14 = sigmoid(-0.817559 * x0 + 0.57470256 * x1 + 0.10937982)  # layer 1 neuron 14
    h1_15 = sigmoid(-2.9374385 * x0 - 0.4316627 * x1 + 1.5588241)  # layer 1 neuron 15
    return -0.5613701 * h1_0 - 0.26400828 * h1_1 + 0.4234089 * h1_2 + 0.23860954 * h1_3 + 0.49227905 * h1_4 + 0.29748696 * h1_5 - 0.70203316 * h1_6 - 0.5406512 * h1_7 - 0.61658525 * h1_8 + 0.46172112 * h1_9 - 0.5101377 * h1_10 + 0.6312087 * h1_11 + 0.5266054 * h1_12 - 0.51801974 * h1_13 - 0.0959294 * h1_14 + 0.5621363 * h1_15 + 0.15405838
