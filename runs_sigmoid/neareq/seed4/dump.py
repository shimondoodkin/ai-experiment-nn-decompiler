# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-7.2297535 * x0 + 7.2201376 * x1 + 1.6592407)  # layer 1 neuron 0
    h1_1 = sigmoid(-11.0711 * x0 + 11.107881 * x1 - 2.3072295)  # layer 1 neuron 1
    h1_2 = sigmoid(-11.009507 * x0 + 11.049075 * x1 - 2.297359)  # layer 1 neuron 2
    h1_3 = sigmoid(11.97291 * x0 - 11.954919 * x1 - 2.382195)  # layer 1 neuron 3
    h1_4 = sigmoid(0.2731691 * x0 + 0.67417383 * x1 - 1.7762611)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.73122567 * x0 - 0.18161269 * x1 - 0.051289845)  # layer 1 neuron 5
    h1_6 = sigmoid(-4.334327 * x0 + 4.3353477 * x1 + 1.3459431)  # layer 1 neuron 6
    h1_7 = sigmoid(-4.969748 * x0 + 4.9679775 * x1 + 1.4074155)  # layer 1 neuron 7
    return 4.6309066 * h1_0 - 11.063248 * h1_1 - 10.155217 * h1_2 - 12.850809 * h1_3 + 0.8635674 * h1_4 + 0.55362684 * h1_5 + 2.4748213 * h1_6 + 2.9036617 * h1_7 + 0.49948505
