# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-0.76792204 * x0 - 0.2158096)  # layer 1 neuron 0
    h1_1 = tanh(-0.09857621 * x0 + 1.0117468)  # layer 1 neuron 1
    h1_2 = tanh(0.071555406 * x0 - 1.0662186)  # layer 1 neuron 2
    h1_3 = tanh(0.09333333 * x0 - 1.0385823)  # layer 1 neuron 3
    h1_4 = tanh(0.08775097 * x0 - 1.4029475)  # layer 1 neuron 4
    h1_5 = tanh(0.1403371 * x0 - 0.54271215)  # layer 1 neuron 5
    h1_6 = tanh(0.6115186 * x0 - 0.48221844)  # layer 1 neuron 6
    h1_7 = tanh(1.1453334 * x0 + 1.2704707)  # layer 1 neuron 7
    return 0.7726639 * h1_0 + 1.1334687 * h1_1 - 1.2358537 * h1_2 - 1.3397155 * h1_3 - 0.4857187 * h1_4 - 0.27599552 * h1_5 - 0.70031744 * h1_6 - 1.3414432 * h1_7 + 0.5430278
