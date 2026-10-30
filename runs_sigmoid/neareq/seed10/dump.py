# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-8.651956 * x0 + 8.672822 * x1 - 2.1071463)  # layer 1 neuron 0
    h1_1 = sigmoid(-10.312301 * x0 + 10.325284 * x1 - 2.3157148)  # layer 1 neuron 1
    h1_2 = sigmoid(-3.6776795 * x0 + 3.6840465 * x1 + 1.2994237)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.2166685 * x0 + 1.5880206 * x1 + 1.9573488)  # layer 1 neuron 3
    h1_4 = sigmoid(11.280688 * x0 - 11.263236 * x1 - 2.2254317)  # layer 1 neuron 4
    h1_5 = sigmoid(-1.1249382 * x0 + 0.52569675 * x1 + 0.1770337)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.6984441 * x0 + 0.94802475 * x1 + 0.48111168)  # layer 1 neuron 6
    h1_7 = sigmoid(0.022561405 * x0 + 0.72445107 * x1 - 1.4393774)  # layer 1 neuron 7
    return -10.5712 * h1_0 - 13.311974 * h1_1 + 3.1556997 * h1_2 + 1.6563127 * h1_3 - 17.090887 * h1_4 + 2.8393066 * h1_5 + 2.579952 * h1_6 + 1.9563072 * h1_7 + 2.4918048
