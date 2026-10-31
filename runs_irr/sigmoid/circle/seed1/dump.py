# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-0.041755784 * x0 - 0.0046558147 * x1 + 3.8199408)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.7189752 * x0 + 4.254968 * x1 - 7.9865746)  # layer 1 neuron 1
    h1_2 = sigmoid(-4.4148593 * x0 - 0.13652235 * x1 - 7.603268)  # layer 1 neuron 2
    h1_3 = sigmoid(1.1820043 * x0 + 4.3242335 * x1 - 7.789005)  # layer 1 neuron 3
    h1_4 = sigmoid(4.0577903 * x0 - 2.6719022 * x1 + 8.476107)  # layer 1 neuron 4
    h1_5 = sigmoid(3.380867 * x0 - 2.9106317 * x1 - 7.925007)  # layer 1 neuron 5
    h1_6 = sigmoid(3.6239588 * x0 - 2.8257518 * x1 + 7.884893)  # layer 1 neuron 6
    h1_7 = sigmoid(3.2070835 * x0 + 2.7880538 * x1 - 7.298324)  # layer 1 neuron 7
    h1_8 = sigmoid(-0.041131657 * x0 - 0.0046580834 * x1 + 4.26917)  # layer 1 neuron 8
    h1_9 = sigmoid(4.3557253 * x0 - 0.6443499 * x1 - 7.7044487)  # layer 1 neuron 9
    h1_10 = sigmoid(-4.313234 * x0 - 1.7326838 * x1 + 8.174897)  # layer 1 neuron 10
    h1_11 = sigmoid(-0.16849142 * x0 - 4.581667 * x1 - 7.9961495)  # layer 1 neuron 11
    h1_12 = sigmoid(-3.5320816 * x0 - 2.4972358 * x1 - 7.6702385)  # layer 1 neuron 12
    h1_13 = sigmoid(2.163007 * x0 + 4.278416 * x1 + 8.12442)  # layer 1 neuron 13
    h1_14 = sigmoid(-0.042532887 * x0 - 0.0047557047 * x1 + 3.534607)  # layer 1 neuron 14
    h1_15 = sigmoid(-1.6273576 * x0 + 4.0987334 * x1 + 7.5336437)  # layer 1 neuron 15
    return -2.5834792 * h1_0 - 11.610167 * h1_1 - 11.723526 * h1_2 - 11.514047 * h1_3 + 5.338558 * h1_4 - 11.281487 * h1_5 + 5.357706 * h1_6 - 10.549851 * h1_7 - 2.466972 * h1_8 - 11.217057 * h1_9 + 5.4312882 * h1_10 - 11.497865 * h1_11 - 11.801744 * h1_12 + 5.629386 * h1_13 - 2.440336 * h1_14 + 5.413549 * h1_15 - 2.291661
