# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.8951028 * x0 + 0.43288788)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.8194047 * x0 + 0.7756699)  # layer 1 neuron 1
    h1_2 = sigmoid(-2.0714734 * x0 - 2.1129053)  # layer 1 neuron 2
    h1_3 = sigmoid(-2.7335036 * x0 - 2.8721933)  # layer 1 neuron 3
    h1_4 = sigmoid(-1.678026 * x0 - 1.4739171)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.8855602 * x0 - 0.087760895)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.3068606 * x0 - 0.72609323)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.7833508 * x0 + 0.56953967)  # layer 1 neuron 7
    return 0.8922549 * h1_0 + 0.81782544 * h1_1 + 0.6975431 * h1_2 + 0.8444522 * h1_3 + 1.1820439 * h1_4 + 1.0575386 * h1_5 + 0.78997266 * h1_6 + 0.8007059 * h1_7 + 0.27767083
