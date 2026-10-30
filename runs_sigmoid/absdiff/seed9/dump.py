# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(0.6867917 * x0 - 0.6795227 * x1 - 2.3895454)  # layer 1 neuron 0
    h1_1 = sigmoid(1.5151458 * x0 - 1.5194346 * x1 + 1.608809)  # layer 1 neuron 1
    h1_2 = sigmoid(1.8771695 * x0 - 1.8751698 * x1 - 1.2455642)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.5688447 * x0 + 1.6127497 * x1 - 6.368917)  # layer 1 neuron 3
    h1_4 = sigmoid(0.114867546 * x0 - 0.20277806 * x1 + 0.6163985)  # layer 1 neuron 4
    h1_5 = sigmoid(0.014017276 * x0 - 0.07441723 * x1 + 3.2873898)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.112018965 * x0 - 0.22183447 * x1 + 1.8809272)  # layer 1 neuron 6
    h1_7 = sigmoid(0.49862143 * x0 - 0.28188646 * x1 + 0.09273112)  # layer 1 neuron 7
    return 5.6083136 * h1_0 - 4.1079564 * h1_1 + 1.7180233 * h1_2 + 3.1936557 * h1_3 + 0.81589234 * h1_4 + 1.2121289 * h1_5 + 0.61548203 * h1_6 + 0.6533365 * h1_7 + 0.24574599
