# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-7.4111724 * x0 - 6.7030153 * x1 + 1.7788588)  # layer 1 neuron 0
    h1_1 = sigmoid(9.363824 * x0 - 6.6941648 * x1 + 2.7177331)  # layer 1 neuron 1
    h1_2 = sigmoid(9.429407 * x0 - 7.4770975 * x1 + 3.3560874)  # layer 1 neuron 2
    h1_3 = sigmoid(7.592143 * x0 - 10.596995 * x1 - 2.6465936)  # layer 1 neuron 3
    return -5.591682 * h1_0 - 8.926418 * h1_1 - 8.828092 * h1_2 + 16.712818 * h1_3 + 9.432862
