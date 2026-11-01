# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-4.2521477 * x0 + 0.7693604)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.672598 * x0 + 0.46885988)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.88853854 * x0 + 0.49497005)  # layer 1 neuron 2
    h1_3 = sigmoid(-3.1228216 * x0 + 0.38000944)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.46142817 * x0 - 0.10332245)  # layer 1 neuron 4
    h1_5 = sigmoid(-5.2515945 * x0 + 1.3674923)  # layer 1 neuron 5
    h1_6 = sigmoid(0.54870665 * x0 - 0.44668555)  # layer 1 neuron 6
    h1_7 = sigmoid(0.5737018 * x0 - 0.4237501)  # layer 1 neuron 7
    return 1.1294345 * h1_0 + 0.92520785 * h1_1 + 0.86803067 * h1_2 + 1.231106 * h1_3 + 0.5009587 * h1_4 + 1.4210764 * h1_5 + 0.35079738 * h1_6 - 0.23809963 * h1_7 + 0.057476893
